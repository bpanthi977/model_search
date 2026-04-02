"""
Visualize Activations Script

This script visualizes the intermediate layer activations of a trained PyTorch model.
It runs a forward pass on validation data, intercepts the outputs of leaf modules
(like nn.Linear, nn.ReLU) using PyTorch forward hooks, and plots the distributions
of these activations per neuron using Seaborn violin plots.

Key Features:
- Automatically splits wide layers (>32 neurons) into a grid of subplots for readability.
- Shares the Y-axis scale across all subplots of a layer for accurate visual comparison.
- Filters out extreme outlier activations per neuron (e.g., keeping only the middle 95%).
- Supports running inference over multiple batches or the entire validation dataset.

Usage Examples:
    # Basic usage (1 batch):
    python visualize_activations.py --checkpoint logs/your_model/

    # Process 2 batches and change batch size:
    python visualize_activations.py --checkpoint logs/your_model/ --dataset.subset 2 --train.batch_size 1024

    # Process ALL validation data and change the outlier filter to 99%:
    python visualize_activations.py --checkpoint logs/your_model/ --dataset.subset -1 --percentile 99.0

Arguments:
    --checkpoint        : Path to the training checkpoint directory containing config.yaml and checkpoint.pth
    --percentile        : Middle percentile of values to show per neuron (default: 95.0). Set to 100.0 to disable.
    --dataset.subset    : Number of batches to process (default: 1). Set to -1 to use all validation data.
    --train.batch_size  : Override the batch size from the config for inference.
    --dataset.sample    : Fraction of the dataset to use (e.g. 0.5).
"""

import argparse
import re
import sys
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import pyrallis
import random
import numpy as np

from config import Config
from model import create_model, Split, SplitInterleave, Join
from dataset import load_dataset

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_leaf(module):
    return len(list(module.children())) == 0

def main():
    parser = argparse.ArgumentParser(
        prog="Visualize Activations",
        description="Visualize model layer activations using Violin plots"
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--percentile", type=float, default=95.0, help="Middle percentile of values to show (e.g. 95.0)")
    # We parse the known args. The rest goes to pyrallis to override config.yaml.
    args, args_rest = parser.parse_known_args()

    # Default to 1 batch if user didn't explicitly override dataset.subset
    if '--dataset.subset' not in args_rest:
        args_rest.extend(['--dataset.subset', '1'])

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint path doesn't exist: {checkpoint_path}")
        exit(1)

    if checkpoint_path.is_file():
        m = re.fullmatch(r'checkpoint(.*?)\.pth', checkpoint_path.name)
        if not m:
            print(f"Checkpoint file name must match 'checkpoint*.pth', got: {checkpoint_path.name}")
            exit(1)
        run_dir = checkpoint_path.parent
        ckpt_file = checkpoint_path
        activations_suffix = m.group(1)   # e.g. "" / "_best" / "-100"
    else:
        run_dir = checkpoint_path
        ckpt_file = run_dir / 'checkpoint.pth'
        activations_suffix = ''

    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        print(f"Config path doesn't exist: {config_path}")
        exit(1)

    try:
        config = pyrallis.parse(config_class=Config, config_path=config_path, args=args_rest)
    except pyrallis.ParsingError as e:
        print(f"Failure to parse config file. Details:\n{e}")
        exit(1)

    print(f"Using {config.dataset.subset} batches for visualization (where -1 means all validation data).")
    print(f"Batch size: {config.train.batch_size}")
    print(f"Dataset sample fraction: {config.dataset.sample}")

    set_all_seeds(42)

    # 1. Load Dataset
    dataset = load_dataset(config.dataset)

    # 2. Create Model
    model = create_model(config.train, dataset)
    
    # 3. Load weights
    if not ckpt_file.exists():
        print(f"Checkpoint file doesn't exist: {ckpt_file}")
        exit(1)
        
    checkpoint_data = torch.load(ckpt_file)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"Model loaded. Best val loss was: {checkpoint_data.get('best_val_loss', 'Unknown')}")

    # 4. Hook Setup
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            
            # Extract just the tensor if output is a tuple
            if isinstance(output, tuple):
                val = output[0].detach().cpu()
            else:
                val = output.detach().cpu()
                
            activations[name].append(val)
        return hook

    # Pre-scan to find computation layers followed by an activation
    _act_types = (nn.ReLU, nn.Tanh, nn.LeakyReLU)
    _skip_types = (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, Split, SplitInterleave, Join)
    _layers_before_act = set()

    for seq_name, seq_mod in model.named_modules():
        if isinstance(seq_mod, nn.Sequential):
            children = list(seq_mod.named_children())
            for i, (cname, cmod) in enumerate(children):
                if isinstance(cmod, _act_types) or isinstance(cmod, _skip_types):
                    continue
                # Check if followed by an activation within next 2 positions
                for j in range(i + 1, min(i + 3, len(children))):
                    next_mod = children[j][1]
                    if isinstance(next_mod, _act_types):
                        full_name = f"{seq_name}.{cname}" if seq_name else cname
                        _layers_before_act.add(full_name)
                        break
                    elif not isinstance(next_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        break  # Hit another computation layer, stop

    # Register hooks for leaf modules, skipping computation layers before activations
    for name, module in model.named_modules():
        if (name != ""
                and is_leaf(module)
                and name not in _layers_before_act
                and not isinstance(module, _skip_types)):
            module.register_forward_hook(get_activation(name))

    # 5. Dataloader Inference
    val_dataset = TensorDataset(dataset.validateX, dataset.validateY)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    device = model.get_device()
    model.eval()

    num_batches_to_run = config.dataset.subset
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(val_dataloader, desc="Running Inference")):
            if num_batches_to_run != -1 and i >= num_batches_to_run:
                break
            x = x.to(device)
            model(x)

    # 6. Plotting
    out_dir = run_dir / "figs" / f"activations{activations_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the command to cmd.sh for reproducibility
    import os
    cmd_script = out_dir / "cmd.sh"
    with open(cmd_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("python " + " ".join(sys.argv) + "\n")
    os.chmod(cmd_script, 0o755)

    import math

    print("Generating violin plots...")
    for name, acts in activations.items():
        if len(acts) == 0:
            continue
            
        # acts is a list of tensors
        cat_acts = torch.cat(acts, dim=0).numpy()
        
        # We handle up to 2D tensors (batch, neurons)
        # If it's a 3D tensor (e.g. sequence), we flatten batch and sequence
        if len(cat_acts.shape) > 2:
            cat_acts = cat_acts.reshape(-1, cat_acts.shape[-1])
            
        num_neurons = cat_acts.shape[1] if len(cat_acts.shape) > 1 else 1

        if num_neurons == 1:
            cat_acts = cat_acts.reshape(-1, 1)

        # Compute per-neuron summary statistics on raw (unfiltered) activations
        raw_acts = cat_acts
        stats_df = pd.DataFrame({
            'neuron': [f"N{j}" for j in range(num_neurons)],
            'mean': np.mean(raw_acts, axis=0),
            'min': np.min(raw_acts, axis=0),
            'max': np.max(raw_acts, axis=0),
            'std': np.std(raw_acts, axis=0),
        })

        # Apply percentile filtering per neuron
        if args.percentile < 100.0:
            lower_bound = np.percentile(cat_acts, (100.0 - args.percentile) / 2.0, axis=0)
            upper_bound = np.percentile(cat_acts, 100.0 - (100.0 - args.percentile) / 2.0, axis=0)

            # Mask outliers with NaN
            mask = (cat_acts >= lower_bound) & (cat_acts <= upper_bound)
            cat_acts = np.where(mask, cat_acts, np.nan)
            
        # Subplot calculation
        max_neurons_per_row = 32
        num_rows = math.ceil(num_neurons / max_neurons_per_row)
        
        # Calculate figure size: max 32 neurons wide
        fig_width = max(10, min(num_neurons, max_neurons_per_row) / 3.0)
        fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, 5 * num_rows), sharey=True)
        
        # Ensure axes is iterable even if num_rows == 1
        if num_rows == 1:
            axes = [axes]
            
        for i in range(num_rows):
            ax = axes[i]
            
            start_idx = i * max_neurons_per_row
            end_idx = min((i + 1) * max_neurons_per_row, num_neurons)
            
            subset_acts = cat_acts[:, start_idx:end_idx]
            
            # Create DataFrame for Seaborn
            df = pd.DataFrame(subset_acts, columns=[f"N{j}" for j in range(start_idx, end_idx)])
            
            # Plot
            sns.violinplot(data=df, linewidth=0.8, density_norm='width', ax=ax)
            
            ax.set_xlabel("Neuron Index")
            ax.set_ylabel("Activation")
            
            # Add horizontal grid lines at tick values for easier reading
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            if (end_idx - start_idx) > 20:
                ax.tick_params(axis='x', rotation=45)
        
        # Set overall title
        total_samples = len(cat_acts)
        
        title = f"Activations for {name} ({total_samples} samples)"
        if args.percentile < 100.0:
            title += f"\nMiddle {args.percentile}% values shown"
            
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        safe_name = name.replace(".", "_")
        if not safe_name: safe_name = "output"
        out_path = out_dir.joinpath(f"{safe_name}.png")
        plt.savefig(out_path)
        plt.savefig(out_dir.joinpath(f"{safe_name}.pdf"))
        plt.savefig(out_dir.joinpath(f"{safe_name}.svg"))
        plt.close()
        print(f"Saved {out_path}")

        csv_path = out_dir.joinpath(f"{safe_name}.csv")
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

if __name__ == "__main__":
    main()
