import torch
import os
from torch.utils.data import random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import argparse
import random
import time
import logging

from models import DiT_Flow_Matching
from diffusion import create_diffusion
from dataset import DiffusionDataset

def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def save_best_checkpoint(model, ema, optimizer, epoch, best_loss, args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get architecture parameters from the model
    hidden_size = model.module.hidden_size if hasattr(model.module, 'hidden_size') else args.hidden_size
    num_heads = model.module.num_heads if hasattr(model.module, 'num_heads') else args.num_heads
    depth = model.module.depth if hasattr(model.module, 'depth') else args.depth
    
    checkpoint_path = os.path.join(save_dir, f"d{hidden_size}_h{num_heads}_bs{args.global_batch_size}_lr{args.lr}_best.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'batch_size': args.global_batch_size,
        'learning_rate': args.lr,
        'architecture': {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'depth': depth
        }
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} with loss {best_loss:.6f}")

def create_logger(logging_dir):
    if get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def register_gradient_hooks(model):
    """Register hooks to track gradient statistics per layer."""
    gradient_stats = {}
    
    def hook_fn(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0]
                gradient_stats[f"grad_norm/{name}"] = grad.norm(2).item()
                gradient_stats[f"grad_max/{name}"] = grad.abs().max().item()
                gradient_stats[f"grad_mean/{name}"] = grad.mean().item()
        return hook
    
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.LayerNorm)):
            layer.register_full_backward_hook(hook_fn(name))
    
    return gradient_stats

def main(args):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    device = torch.device(f'cuda:{local_rank}')
    print(f"Using device: {device}")

    setup_ddp()
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------
    # Setup experiment
    # ------------------------------
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = random.randint(0, 100)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-DiT"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ------------------------------
    # Load Dataset
    # ------------------------------
    dataset = DiffusionDataset(
        latent_features_path=args.latent_features_path,
        node_features_path=args.node_features_path,
        edge_index_path=args.edge_index_path,
        curves_path=args.curves_path,
        max_nodes=args.max_nodes
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True, 
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=int(args.global_batch_size // world_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, 
        sampler=val_sampler, 
        batch_size=int(args.global_batch_size // world_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Train dataset contains {len(train_dataset):,} samples")
    logger.info(f"Val dataset contains {len(val_dataset):,} samples")

    # ------------------------------
    # Model, Optimizer
    # ------------------------------
    model = DiT_Flow_Matching(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        class_dropout_prob=args.class_dropout_prob,
        learn_sigma=args.learn_sigma,
        seq_len=args.seq_len,
    ).to(device)
    
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[local_rank])

    # Register gradient hooks (rank 0 only)
    if rank == 0:
        gradient_stats = register_gradient_hooks(model.module)
    else:
        gradient_stats = {}
        
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(timestep_respacing="")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize WandB after model creation
    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=f"d{args.hidden_size}_h{args.num_heads}_bs{args.global_batch_size}_lr{args.lr}",
            config={
                # Architecture
                "input_size": args.input_size,
                "hidden_size": args.hidden_size,
                "num_heads": args.num_heads,
                "depth": args.depth,
                "mlp_ratio": args.mlp_ratio,
                "class_dropout_prob": args.class_dropout_prob,
                "learn_sigma": args.learn_sigma,
                "seq_len": args.seq_len,
                
                # Training
                "global_batch_size": args.global_batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "global_seed": args.global_seed
            },
            dir=experiment_dir
        )

    best_loss = float("inf")
    update_ema(ema, model.module, decay=0)  # Initialize EMA

    for epoch in range(args.epochs):
        # Training
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kl = 0.0

        with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
            for batch_idx, batch in enumerate(train_loader):  # Add enumerate here                
                x = batch['node_features'].to(device)
                curves = batch['curve_data'].to(device)
                adj = batch['adjacency'].to(device)
                
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(curves=curves, adj=adj)
                
                loss_dict = diffusion.training_losses_flow_matching(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                update_ema(ema, model.module)

                train_loss += loss.item()
                if "mse" in loss_dict:
                    train_mse += loss_dict["mse"].mean().item()
                if "vb" in loss_dict:
                    train_kl += loss_dict["vb"].mean().item()

                if rank == 0 and (batch_idx + 1) % args.log_every == 0:
                    wandb.log({
                        **gradient_stats,
                        "train/loss": loss.item(),
                        "train/mse": loss_dict["mse"].mean().item() if "mse" in loss_dict else 0,
                        "train/kl": loss_dict["vb"].mean().item() if "vb" in loss_dict else 0,
                        "epoch": epoch,
                        "batch": batch_idx
                    })

                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mse = train_mse / len(train_loader)
        avg_train_kl = train_kl / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_kl = 0.0

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Rank {rank} Val Epoch {epoch}", leave=False) as pbar:
                for batch in val_loader:
                    x = batch['node_features'].to(device)
                    curves = batch['curve_data'].to(device)
                    adj = batch['adjacency'].to(device)
                    
                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                    model_kwargs = dict(curves=curves, adj=adj)
                    
                    loss_dict = diffusion.training_losses_flow_matching(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

                    val_loss += loss.item()
                    if "mse" in loss_dict:
                        val_mse += loss_dict["mse"].mean().item()
                    if "vb" in loss_dict:
                        val_kl += loss_dict["vb"].mean().item()

                    if rank == 0:
                        wandb.log({
                            "val/loss": loss.item(),
                            "val/mse": loss_dict["mse"].mean().item() if "mse" in loss_dict else 0,
                            "val/kl": loss_dict["vb"].mean().item() if "vb" in loss_dict else 0,
                            "epoch": epoch,
                        })

                    pbar.set_postfix({"Val Loss": loss.item()})
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)

        if rank == 0:
            logger.info(
                f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f}, KL: {avg_train_kl:.4f}) | "
                f"Val Loss: {avg_val_loss:.4f} (MSE: {avg_val_mse:.4f}, KL: {avg_val_kl:.4f})"
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_loss=best_loss,
                    args=args,
                    save_dir=checkpoint_dir
                )

        # Save checkpoint periodically
        if (epoch + 1) % args.ckpt_every == 0 and rank == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'args': args
            }, checkpoint_path)
            logger.info(f"Saved periodic checkpoint at {checkpoint_path}")

    if rank == 0:
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Training arguments
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--wandb-project", type=str, default="DiT-Graph-Diffusion")
    
    # Model architecture arguments
    parser.add_argument("--input-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--learn-sigma", type=bool, default=True)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=10)
    
    # Dataset paths
    parser.add_argument("--latent-features-path", type=str, 
                      default="/home/anurizada/Documents/nobari_10_joints/structured_latent_features.npy")
    parser.add_argument("--node-features-path", type=str, 
                      default="/home/anurizada/Documents/nobari_10_joints/node_features.npy")
    parser.add_argument("--edge-index-path", type=str, 
                      default="/home/anurizada/Documents/nobari_10_joints/edge_index.npy")
    parser.add_argument("--curves-path", type=str, 
                      default="/home/anurizada/Documents/nobari_10_joints/curves.npy")
    
    args = parser.parse_args()
    main(args)