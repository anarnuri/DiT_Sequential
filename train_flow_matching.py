import torch
import torch.nn as nn 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import argparse
import logging
import os
import time
import wandb
from tqdm import tqdm
import random

from models import DiT_Flow_Matching
from diffusion import create_diffusion
from dataset import DiffusionDataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
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

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    
    # Setup DDP
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = random.randint(0, 100)
        model_string_name = 'DiT'
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # Initialize wandb
        wandb.init(
            project="DiT-Graph-Diffusion",
            name=f"run-{experiment_index:03d}",
            config=vars(args),
            dir=experiment_dir
        )
    else:
        logger = create_logger(None)

    # Create model
    model = DiT_Flow_Matching(
        input_size=2,
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        seq_len=10,
    )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    
    # Register gradient hooks
    if rank == 0:
        gradient_stats = register_gradient_hooks(model.module)
    else:
        gradient_stats = {}

    # Configure diffusion
    diffusion = create_diffusion(timestep_respacing="")

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    dataset = DiffusionDataset(
        latent_features_path='/home/anurizada/Documents/nobari_10_joints/structured_latent_features.npy',
        node_features_path='/home/anurizada/Documents/nobari_10_joints/node_features.npy',
        edge_index_path='/home/anurizada/Documents/nobari_10_joints/edge_index.npy',
        curves_path='/home/anurizada/Documents/nobari_10_joints/curves.npy',
        max_nodes=10
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(dataset):,} samples")

    # Prepare models for training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Training variables
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_mse = 0
    running_kl = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", disable=rank != 0)
        for batch in progress_bar:
            # Load and prepare batch data
            x = batch['node_features'].to(device)
            curves = batch['curve_data'].to(device)
            adj = batch['adjacency'].to(device)
            
            # Diffusion process
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(curves=curves, adj=adj)
            
            # Get loss components
            loss_dict = diffusion.training_losses_flow_matching(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            # Backpropagation
            opt.zero_grad()
            loss.backward()
            
            # Log gradients (rank 0 only)
            if rank == 0 and train_steps % args.log_every == 0:
                wandb.log(gradient_stats, step=train_steps)

            opt.step()
            update_ema(ema, model.module)

            # Logging
            running_loss += loss.item()
            if "mse" in loss_dict:
                running_mse += loss_dict["mse"].mean().item()
            if "vb" in loss_dict:
                running_kl += loss_dict["vb"].mean().item()
            log_steps += 1
            train_steps += 1

            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

            # Log metrics periodically
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce metrics across processes
                metrics = {
                    "loss": running_loss / log_steps,
                    "mse": running_mse / log_steps if "mse" in loss_dict else 0,
                    "vb": running_kl / log_steps if "vb" in loss_dict else 0
                }
                
                reduced_metrics = {}
                for name, value in metrics.items():
                    tensor = torch.tensor(value, device=device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    reduced_metrics[name] = tensor.item() / world_size
                
                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {reduced_metrics['loss']:.4f}, "
                    f"MSE: {reduced_metrics['mse']:.4f}, "
                    f"KL: {reduced_metrics['vb']:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )
                
                if rank == 0:
                    wandb.log({
                        "train/loss": reduced_metrics['loss'],
                        "train/mse": reduced_metrics['mse'],
                        "train/kl": reduced_metrics['vb'],
                        "train/steps_per_sec": steps_per_sec,
                        "train/step": train_steps,
                        "train/epoch": epoch
                    }, step=train_steps)

                # Reset metrics
                running_loss = 0
                running_mse = 0
                running_kl = 0
                log_steps = 0
                start_time = time.time()
            
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                # All ranks wait here before checkpoint starts
                dist.barrier()
                
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "step": train_steps,
                        "epoch": epoch
                    }
                    # Atomic save
                    tmp_path = f"{checkpoint_dir}/{train_steps:07d}.pt.tmp"
                    torch.save(checkpoint, tmp_path)
                    os.rename(tmp_path, f"{checkpoint_dir}/{train_steps:07d}.pt")
                    logger.info(f"Saved checkpoint")
                
                # All ranks wait here after checkpoint completes
                dist.barrier()

                # Cleanup
                if rank == 0:
                    wandb.finish()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Training arguments
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1000,
                       help="Log metrics every N steps")
    parser.add_argument("--ckpt-every", type=int, default=50_000,
                       help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    main(args)
