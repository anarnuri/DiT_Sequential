import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
from torchvision import models
from utils import preprocess_curves

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class SequenceEmbedder(nn.Module):

    def __init__(self, in_dim, hidden_dim, seq_len, use_norm=True, activation=nn.GELU, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.residual = residual and (in_dim == hidden_dim)
        self.seq_len = seq_len
        
        self.proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.activation = activation() if activation is not None else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, in_dim)
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        identity = x if self.residual else 0
        x = self.proj(x)
        x = self.activation(x)
        x = self.norm(x)
        return x + identity if self.residual else x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds continuous label tensors into vector representations. 
    Handles dropout for classifier-free guidance.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        # Projection layer instead of embedding table
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # For classifier-free guidance dropout
        self.null_embed = nn.Parameter(torch.randn(hidden_size))
        
    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        Returns either original labels or null embeddings.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        # Expand null embeddings to match batch size
        null_embeds = self.null_embed.unsqueeze(0).repeat(labels.shape[0], 1)
        return torch.where(drop_ids.unsqueeze(-1), null_embeds, labels)

    def forward(self, labels, train, force_drop_ids=None):
        """
        Args:
            labels: (batch, dim) tensor of continuous labels
            train: bool indicating training mode
            force_drop_ids: optional tensor to force certain samples to drop labels
            
        Returns:
            (batch, hidden_size) tensor of embedded labels
        """
        # Project input labels to hidden size
        embeddings = self.proj(labels)
        
        # Apply classifier-free dropout if needed
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            embeddings = self.token_drop(embeddings, force_drop_ids)
            
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, use_ln=True):
        super().__init__()
        # First projection
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_ln else nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.layer_scale1 = nn.Parameter(torch.ones(hidden_dim))
        
        # Second projection (bottleneck)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_ln else nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)
        self.layer_scale2 = nn.Parameter(torch.ones(hidden_dim))
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        # First projection
        x = self.fc1(x)
        x = self.norm1(x) if x.ndim == 2 else self.norm1(x.unsqueeze(0)).squeeze(0)
        x = self.act1(x) * self.layer_scale1
        x = self.dropout1(x)
        
        # Second projection
        x = self.fc2(x)
        x = self.norm2(x) if x.ndim == 2 else self.norm2(x.unsqueeze(0)).squeeze(0)
        x = self.act2(x) * self.layer_scale2
        x = self.dropout2(x)
        
        # Output projection
        x = self.fc_out(x)
        
        return x
    
class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels=1, emb_size=128):
        super().__init__()
        self.convnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.convnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convnet.fc = nn.Identity()
        self.projector = ProjectionHead(2048, 2048, emb_size)

    def forward(self, x):
        features = self.convnet(x)
        return self.projector(features)
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=256,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        seq_len=10,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_dims = input_size
        self.out_dims = input_size
        self.num_heads = num_heads

        self.x_embedder = SequenceEmbedder(input_size, hidden_size, seq_len=seq_len)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ConditionEmbedder(hidden_size, hidden_size, class_dropout_prob)
        
        # Add Contrastive_Curve and GraphHop models
        self.contrastive_curve = ContrastiveEncoder(in_channels=1, emb_size=hidden_size//2)
        self.contrastive_adj = ContrastiveEncoder(in_channels=1, emb_size=hidden_size//2)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer_mean = FinalLayer(hidden_size, self.out_dims)
        self.final_layer_var = FinalLayer(hidden_size, self.out_dims)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize sequence embedder
        if hasattr(self.x_embedder, 'proj'):
            if isinstance(self.x_embedder.proj, nn.Linear):
                nn.init.xavier_uniform_(self.x_embedder.proj.weight)
                if self.x_embedder.proj.bias is not None:
                    nn.init.constant_(self.x_embedder.proj.bias, 0)
            elif isinstance(self.x_embedder.proj, nn.Conv1d):
                nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.x_embedder.proj.weight.shape[0], -1))
                if self.x_embedder.proj.bias is not None:
                    nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if hasattr(self.y_embedder, 'embedding_table'):
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        elif hasattr(self.y_embedder, 'proj'):
            # Initialize each layer in the Sequential proj
            for layer in self.y_embedder.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Initialize timestep embedding MLP:
        if hasattr(self.t_embedder, 'mlp'):
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            if hasattr(self.t_embedder.mlp[0], 'bias') and self.t_embedder.mlp[0].bias is not None:
                nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
            if hasattr(self.t_embedder.mlp[2], 'bias') and self.t_embedder.mlp[2].bias is not None:
                nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        for final_layer in [self.final_layer_mean, self.final_layer_var]:
            if hasattr(final_layer, 'adaLN_modulation'):
                nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            if hasattr(final_layer, 'linear'):
                nn.init.constant_(final_layer.linear.weight, 0)
                nn.init.constant_(final_layer.linear.bias, 0)

    def forward(self, x, t, **model_kwargs):
        # Extract conditioning data from model_kwargs
        curves = model_kwargs.get('curves')
        adj = model_kwargs.get('adj')

        # 1. Input embedding
        x_embed = self.x_embedder(x)  # (batch, seq_len, hidden_dim)
        x = x_embed + self.pos_embed  # Add positional embedding

        # 2. Time embedding
        t_embed = self.t_embedder(t)  # (batch, hidden_dim)

        # 3. Process conditioning data
        # Preprocess and embed curves
        curves = preprocess_curves(curves).unsqueeze(1)
        curve_embed = self.contrastive_curve(curves)  # (batch, d_model//2)
        
        # Process graph adjacency
        adj_embed = self.contrastive_adj(adj)  # (batch, d_model//2)
        
        # Combine conditioning embeddings
        y = torch.cat([curve_embed, adj_embed], dim=1)  # (batch, d_model)

        y_embed = self.y_embedder(y, self.training)  # (batch, hidden_dim)

        # 4. Combine all conditioning
        c = t_embed + y_embed  # (batch, hidden_dim)

        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 6. Final projection (mean and variance)
        x_mean = self.final_layer_mean(x, c)  # (batch, seq_len, dim)
        x_var = self.final_layer_var(x, c)    # (batch, seq_len, dim)
        
        # Concatenate along sequence length dimension
        x_out = torch.cat([x_mean, x_var], dim=1)  # (batch, seq_len*2, dim)

        return x_out

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiT_Flow_Matching(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=256,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        seq_len=10,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_dims = input_size
        self.out_dims = input_size
        self.num_heads = num_heads

        self.x_embedder = SequenceEmbedder(input_size, hidden_size, seq_len=seq_len)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ConditionEmbedder(hidden_size, hidden_size, class_dropout_prob)
        
        # Add Contrastive_Curve and GraphHop models
        self.contrastive_curve = ContrastiveEncoder(in_channels=1, emb_size=hidden_size//2)
        self.contrastive_adj = ContrastiveEncoder(in_channels=1, emb_size=hidden_size//2)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_dims)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize sequence embedder
        if hasattr(self.x_embedder, 'proj'):
            if isinstance(self.x_embedder.proj, nn.Linear):
                nn.init.xavier_uniform_(self.x_embedder.proj.weight)
                if self.x_embedder.proj.bias is not None:
                    nn.init.constant_(self.x_embedder.proj.bias, 0)
            elif isinstance(self.x_embedder.proj, nn.Conv1d):
                nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.x_embedder.proj.weight.shape[0], -1))
                if self.x_embedder.proj.bias is not None:
                    nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if hasattr(self.y_embedder, 'embedding_table'):
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        elif hasattr(self.y_embedder, 'proj'):
            # Initialize each layer in the Sequential proj
            for layer in self.y_embedder.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Initialize timestep embedding MLP:
        if hasattr(self.t_embedder, 'mlp'):
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            if hasattr(self.t_embedder.mlp[0], 'bias') and self.t_embedder.mlp[0].bias is not None:
                nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
            if hasattr(self.t_embedder.mlp[2], 'bias') and self.t_embedder.mlp[2].bias is not None:
                nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        for final_layer in [self.final_layer]:
            if hasattr(final_layer, 'adaLN_modulation'):
                nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            if hasattr(final_layer, 'linear'):
                nn.init.constant_(final_layer.linear.weight, 0)
                nn.init.constant_(final_layer.linear.bias, 0)

    def forward(self, x, t, **model_kwargs):
        # Extract conditioning data from model_kwargs
        curves = model_kwargs.get('curves')
        adj = model_kwargs.get('adj')

        # 1. Input embedding
        x_embed = self.x_embedder(x)  # (batch, seq_len, hidden_dim)
        x = x_embed + self.pos_embed  # Add positional embedding

        # 2. Time embedding
        t_embed = self.t_embedder(t)  # (batch, hidden_dim)

        # 3. Process conditioning data
        # Preprocess and embed curves
        curves = preprocess_curves(curves).unsqueeze(1)
        curve_embed = self.contrastive_curve(curves)  # (batch, d_model//2)
        
        # Process graph adjacency
        adj_embed = self.contrastive_adj(adj)  # (batch, d_model//2)
        
        # Combine conditioning embeddings
        y = torch.cat([curve_embed, adj_embed], dim=1)  # (batch, d_model)

        y_embed = self.y_embedder(y, self.training)  # (batch, hidden_dim)

        # 4. Combine all conditioning
        c = t_embed + y_embed  # (batch, hidden_dim)

        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 6. Final projection (mean and variance)
        x_out = self.final_layer(x, c)  # (batch, seq_len, dim)
        
        return x_out

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Create 1D sin-cos position embeddings.
    Args:
        embed_dim: embedding dimension (should be even)
        length: length of the sequence
    Returns:
        position embedding of shape (length, embed_dim)
    """
    position = np.arange(length, dtype=np.float64)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, position)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
