"""
Structured pruning script for trained MLP models.

Identifies neurons that never activate (max|activation| <= threshold across the
validation set) and removes them, producing a smaller but functionally equivalent model.

Usage:
    python prune_model.py --checkpoint logs/your_run/
    python prune_model.py --checkpoint logs/your_run/ --threshold 1e-9 --dataset.subset -1
    python prune_model.py --checkpoint logs/your_run/ --output_dir logs/your_run_pruned/
"""

import argparse
import re
import sys
from pathlib import Path

import pyrallis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import Config
from dataset import load_dataset
from model import MULT0, MULT1, NALU, NALUi1, NALUi2
from train import create_model
from utils import get_loss_fn

# Layer types whose output dimension can be pruned
PRUNABLE_TYPES = (nn.Linear, NALU, NALUi1, NALUi2, MULT0, MULT1)


def collect_max_activations(model, dataset, batch_size, num_batches):
    """Run inference and track max|activation| per neuron for each prunable layer."""
    max_acts = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                val = output[0].detach().cpu()
            else:
                val = output.detach().cpu()
            # Flatten group dims (e.g. [batch, groups, features] after Split → [batch*groups, features])
            if val.dim() > 2:
                val = val.reshape(-1, val.shape[-1])
            # Running max per neuron (dim 0 = batch)
            batch_max = val.abs().max(dim=0).values
            if name not in max_acts:
                max_acts[name] = torch.zeros_like(batch_max)
            max_acts[name] = torch.maximum(max_acts[name], batch_max)
        return hook

    _act_types = (nn.ReLU, nn.Tanh, nn.LeakyReLU)
    _skip_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)

    # Map activation layer seq_idx -> preceding prunable layer seq_idx
    act_to_linear = {}
    children = list(model.model._modules.items())  # [(idx_str, module), ...]
    for i, (idx_str, module) in enumerate(children):
        if not isinstance(module, _act_types):
            continue
        # Scan backwards for the nearest preceding prunable layer (skip BN/Dropout)
        for j in range(i - 1, -1, -1):
            prev_idx_str, prev_mod = children[j]
            if isinstance(prev_mod, PRUNABLE_TYPES):
                act_to_linear[int(idx_str)] = int(prev_idx_str)
                break
            elif not isinstance(prev_mod, _skip_types):
                break  # Hit a non-passthrough layer, stop

    linear_with_act = set(act_to_linear.values())

    handles = []
    for idx_str, module in model.model._modules.items():
        seq_idx = int(idx_str)
        if isinstance(module, _act_types) and seq_idx in act_to_linear:
            # Hook activation; key max_acts by the preceding linear's name
            linear_idx = act_to_linear[seq_idx]
            handles.append(module.register_forward_hook(make_hook(f"model.{linear_idx}")))
        elif isinstance(module, PRUNABLE_TYPES) and seq_idx not in linear_with_act:
            # Prunable layer with no following activation (e.g. NALU-I, final output layer)
            handles.append(module.register_forward_hook(make_hook(f"model.{seq_idx}")))

    val_loader = DataLoader(
        TensorDataset(dataset.validateX, dataset.validateY),
        batch_size=batch_size,
        shuffle=False,
    )
    device = model.get_device()
    model.eval()

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(val_loader, desc="Analysing activations")):
            if num_batches != -1 and i >= num_batches:
                break
            model(x.to(device))

    for h in handles:
        h.remove()

    return max_acts


def build_keep_indices(model, max_acts, threshold):
    """
    Compute which output neurons to keep per hidden layer.

    Returns:
        linear_positions: list of Sequential indices for prunable layers (in order)
        keep_output: dict[seq_idx -> 1D LongTensor of keep indices, or None for output layer]
    """
    linear_positions = [
        int(name)
        for name, module in model.model._modules.items()
        if isinstance(module, PRUNABLE_TYPES)
    ]

    keep_output = {}
    for i, seq_idx in enumerate(linear_positions):
        hook_name = f"model.{seq_idx}"
        if i == len(linear_positions) - 1:
            # Output layer: never prune output neurons
            keep_output[seq_idx] = None
        else:
            max_act = max_acts.get(hook_name, torch.zeros(1))
            keep_output[seq_idx] = (max_act > threshold).nonzero(as_tuple=True)[0]

    return linear_positions, keep_output


def _index(tensor, rows=None, cols=None):
    """Index a 1D or 2D tensor by row and/or column keep-indices."""
    if tensor.dim() == 1:
        idx = rows if rows is not None else cols
        return tensor[idx] if idx is not None else tensor
    # 2D
    t = tensor[rows] if rows is not None else tensor
    t = t[:, cols] if cols is not None else t
    return t


def build_pruned_state_dict(model, linear_positions, keep_output):
    """
    Slice every parameter in the state dict according to keep_output indices.
    """
    # Map seq_idx -> keep_input (= keep_output of the previous linear layer,
    # expanded if a Join layer multiplied the feature dimension by n_groups)
    def _out_dim(m):
        if isinstance(m, nn.Linear): return m.weight.shape[0]
        if isinstance(m, NALU): return m.W_hat.shape[0]
        return m.w_hat.shape[1]  # NALUi1/i2/MULT0/MULT1: [in, out]

    def _in_dim(m):
        if isinstance(m, nn.Linear): return m.weight.shape[1]
        if isinstance(m, NALU): return m.W_hat.shape[1]
        return m.w_hat.shape[0]  # NALUi1/i2/MULT0/MULT1: [in, out]

    keep_input = {}
    for i, seq_idx in enumerate(linear_positions):
        if i == 0:
            keep_input[seq_idx] = None
        else:
            prev_seq = linear_positions[i - 1]
            ki = keep_output[prev_seq]
            if ki is not None:
                prev_out = _out_dim(model.model[prev_seq])
                curr_in = _in_dim(model.model[seq_idx])
                # If curr_in > prev_out and divisible, a Join expanded groups
                if curr_in != prev_out and prev_out > 0 and curr_in % prev_out == 0:
                    n_groups = curr_in // prev_out
                    ki = torch.cat([ki + g * prev_out for g in range(n_groups)])
            keep_input[seq_idx] = ki

    # Map BatchNorm seq_idx -> keep indices inherited from the preceding linear layer
    bn_keep = {}
    prev_linear = None
    for idx_str, module in model.model._modules.items():
        seq_idx = int(idx_str)
        if isinstance(module, PRUNABLE_TYPES):
            prev_linear = seq_idx
        elif isinstance(module, nn.BatchNorm1d) and prev_linear is not None:
            bn_keep[seq_idx] = keep_output[prev_linear]

    pruned_sd = {}
    original_sd = model.state_dict()

    for key, tensor in original_sd.items():
        # Keys look like "model.{seq_idx}.{param_name}"
        match = re.match(r'^model\.(\d+)\.(.+)$', key)
        if match is None:
            pruned_sd[key] = tensor
            continue

        seq_idx = int(match.group(1))
        param_name = match.group(2)
        module = model.model[seq_idx]

        if isinstance(module, nn.Linear):
            ki = keep_input[seq_idx]
            ko = keep_output[seq_idx]
            if param_name == 'weight':
                pruned_sd[key] = _index(tensor, rows=ko, cols=ki)
            elif param_name == 'bias':
                pruned_sd[key] = _index(tensor, rows=ko) if ko is not None else tensor
            else:
                pruned_sd[key] = tensor

        elif isinstance(module, NALU):
            # W_hat, M_hat, G: shape [out, in]
            ki = keep_input[seq_idx]
            ko = keep_output[seq_idx]
            if param_name in ('W_hat', 'M_hat', 'G'):
                pruned_sd[key] = _index(tensor, rows=ko, cols=ki)
            else:
                pruned_sd[key] = tensor

        elif isinstance(module, (NALUi1, NALUi2, MULT0, MULT1)):
            # w_hat, m_hat, w_hat1, etc.: shape [in, out]; g: shape [out]
            ki = keep_input[seq_idx]
            ko = keep_output[seq_idx]
            if param_name in ('w_hat', 'm_hat', 'w_hat1', 'm_hat1', 'w_hat2', 'm_hat2'):
                pruned_sd[key] = _index(tensor, rows=ki, cols=ko)
            elif param_name == 'g':
                pruned_sd[key] = _index(tensor, rows=ko) if ko is not None else tensor
            else:
                pruned_sd[key] = tensor

        elif isinstance(module, nn.BatchNorm1d):
            ko = bn_keep.get(seq_idx)
            if param_name == 'num_batches_tracked':
                pruned_sd[key] = tensor  # scalar
            elif ko is not None:
                pruned_sd[key] = _index(tensor, rows=ko)
            else:
                pruned_sd[key] = tensor

        else:
            pruned_sd[key] = tensor

    return pruned_sd


def update_hidden_layers_config(config, linear_positions, keep_output):
    """
    Rewrite config.train.model.hidden_layers with pruned sizes.
    Only modifies layers corresponding to hidden prunable layers.
    """
    # linear_positions[:-1] are the hidden layers; map them in order
    hidden_linear_positions = linear_positions[:-1]
    hl_iter = iter(hidden_linear_positions)

    new_hidden_layers = []
    last_orig_size = None  # original output size of most recent prunable layer
    last_new_size = None   # pruned output size of most recent prunable layer

    for spec in config.train.model.hidden_layers:
        if isinstance(spec, int):
            seq_idx = next(hl_iter)
            new_size = len(keep_output[seq_idx])
            last_orig_size = spec
            last_new_size = new_size
            # Store as string: parse_hidden_layers returns a 2-tuple for plain ints
            # but model.py expects 3-tuples (as produced by the string branch).
            new_hidden_layers.append(str(new_size))
        elif isinstance(spec, str):
            stripped = spec.strip()
            if re.match(r'^(split|split_interleave)', stripped):
                # Reshape-only; no learnable output dimension to update
                new_hidden_layers.append(spec)
            elif re.match(r'^join', stripped):
                # join(N) encodes the flat output size after flattening groups.
                # N = n_groups * neurons_per_group. After pruning the preceding
                # layer from last_orig_size to last_new_size, update N accordingly.
                m = re.search(r'\d+', stripped)
                if m and last_orig_size is not None and last_orig_size > 0:
                    old_n = int(m.group())
                    n_groups = old_n // last_orig_size
                    new_n = n_groups * last_new_size
                    new_spec = re.sub(r'\d+', str(new_n), spec, count=1)
                    new_hidden_layers.append(new_spec)
                else:
                    new_hidden_layers.append(spec)
            else:
                # Has a numeric size — replace first number in the spec
                seq_idx = next(hl_iter)
                new_size = len(keep_output[seq_idx])
                m = re.search(r'\d+', stripped)
                last_orig_size = int(m.group()) if m else None
                last_new_size = new_size
                new_spec = re.sub(r'\d+', str(new_size), spec, count=1)
                new_hidden_layers.append(new_spec)
        else:
            new_hidden_layers.append(spec)

    config.train.model.hidden_layers = new_hidden_layers


def format_summary(model, linear_positions, keep_output):
    lines = []
    lines.append(f"\n{'Layer':<15} {'Original':>10} {'Pruned':>10} {'% Kept':>8}")
    lines.append("-" * 47)
    total_orig = 0
    total_pruned = 0
    for i, seq_idx in enumerate(linear_positions):
        module = model.model[seq_idx]
        if isinstance(module, nn.Linear):
            orig = module.weight.shape[0]
        elif isinstance(module, NALU):
            orig = module.W_hat.shape[0]
        else:  # NALUi1/i2/MULT0/MULT1: [in, out]
            orig = module.w_hat.shape[1]

        ko = keep_output[seq_idx]
        if ko is None:
            label = "(output, fixed)"
            pruned = orig
        else:
            pruned = len(ko)
            label = ""

        pct = 100.0 * pruned / orig if orig > 0 else 0.0
        lines.append(f"model.{seq_idx:<9} {orig:>10} {pruned:>10} {pct:>7.1f}%  {label}")
        total_orig += orig
        total_pruned += pruned

    pct_total = 100.0 * total_pruned / total_orig if total_orig > 0 else 0.0
    lines.append("-" * 47)
    lines.append(f"{'Total':<15} {total_orig:>10} {total_pruned:>10} {pct_total:>7.1f}%")
    lines.append("")
    return "\n".join(lines)


def print_summary(model, linear_positions, keep_output):
    print(format_summary(model, linear_positions, keep_output))


def main():
    parser = argparse.ArgumentParser(
        prog="prune_model",
        description="Structured pruning: remove inactive neurons from a trained MLP.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (contains config.yaml and checkpoint.pth)")
    parser.add_argument("--threshold", type=float, default=1e-9,
                        help="Neurons with max|activation| <= threshold are pruned (default: 1e-9)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for pruned model (default: <checkpoint>_pruned)")
    args, args_rest = parser.parse_known_args()

    # Default: use all validation data
    if '--dataset.subset' not in args_rest:
        args_rest.extend(['--dataset.subset', '-1'])

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint path doesn't exist: {checkpoint_path}")
        sys.exit(1)

    config_path = checkpoint_path / 'config.yaml'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    try:
        config = pyrallis.parse(config_class=Config, config_path=config_path, args=args_rest)
    except pyrallis.ParsingError as e:
        print(f"Failed to parse config: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(str(checkpoint_path) + "_pruned")

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {output_dir}")
    print(f"Threshold  : {args.threshold}")
    print(f"Dataset    : {config.dataset.subset} batches (-1 = all)")

    # Load dataset and model
    dataset = load_dataset(config.dataset)
    model = create_model(config.train, dataset)

    ckpt_file = checkpoint_path / 'checkpoint_best.pth'
    if not ckpt_file.exists():
        ckpt_file = checkpoint_path / 'checkpoint.pth'
    if not ckpt_file.exists():
        print(f"No checkpoint.pth found in {checkpoint_path}")
        sys.exit(1)

    checkpoint_data = torch.load(ckpt_file)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"Loaded checkpoint (best_val_loss={checkpoint_data.get('best_val_loss', 'n/a')})\n")

    # Activation analysis
    max_acts = collect_max_activations(
        model, dataset,
        batch_size=config.train.batch_size,
        num_batches=config.dataset.subset,
    )

    # Compute keep indices
    linear_positions, keep_output = build_keep_indices(model, max_acts, args.threshold)

    if not linear_positions:
        print("No prunable layers found. Exiting.")
        sys.exit(0)

    # Summary
    neuron_summary = format_summary(model, linear_positions, keep_output)
    print(neuron_summary)

    # Build pruned state dict
    pruned_sd = build_pruned_state_dict(model, linear_positions, keep_output)

    # Update config
    orig_hidden_layers = list(config.train.model.hidden_layers)
    update_hidden_layers_config(config, linear_positions, keep_output)

    # Build pruned model and load weights
    pruned_model = create_model(config.train, dataset)
    pruned_model.load_state_dict(pruned_sd)

    # Parity check
    print("Running parity check...")
    pruned_model.eval()
    model.eval()
    val_loader = DataLoader(
        TensorDataset(dataset.validateX, dataset.validateY),
        batch_size=config.train.batch_size,
        shuffle=False,
    )
    device = model.get_device()

    loss_fn = get_loss_fn(config.train.loss)
    total_val_loss = 0.0
    max_diff = 0.0
    n_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            out_orig = model.model(model.normalize(x_batch, model.normalizeX))
            out_pruned = pruned_model.model(pruned_model.normalize(x_batch, pruned_model.normalizeX))
            y_norm = pruned_model.normalize(y_batch, pruned_model.normalizeY)

            total_val_loss += loss_fn(out_pruned, y_norm).item() * batch_size
            max_diff = max(max_diff, (out_orig - out_pruned).abs().max().item())
            n_samples += batch_size

    pruned_val_loss = total_val_loss / n_samples
    print()
    print(f"Pruned model validation loss: {pruned_val_loss}")

    if max_diff <= 1e-10:
        print("Parity check PASSED: outputs match within atol=1e-10.\n")
    else:
        print(f"WARNING: Parity check FAILED (max diff = {max_diff:.2e}). "
              f"This may indicate non-zero neurons were pruned.\n")

    orig_val_loss = checkpoint_data.get('best_val_loss', float('nan'))
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    comparison_table = "\n".join([
        f"{'':20} {'Original':>20} {'Pruned':>20}",
        "-" * 62,
        f"{'Val loss':20} {orig_val_loss:>20.6f} {pruned_val_loss:>20.6f}",
        f"{'Parameters':20} {orig_params:>20,} {pruned_params:>20,}",
        f"{'Hidden layers':20} {str(orig_hidden_layers):>20} {str(config.train.model.hidden_layers):>20}",
        "",
    ])
    print(comparison_table)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    config_out = output_dir / 'config.yaml'
    with open(config_out, 'w') as f:
        f.write(pyrallis.dump(config))
    print(f"Saved config  -> {config_out}")

    threshold_str = f"{args.threshold:g}"

    info_out = output_dir / f'prune_info_t{threshold_str}_.txt'
    dataset_info = "\n".join([
        "Dataset",
        f"  db_file : {config.dataset.db_file}",
        f"  sample  : {config.dataset.sample}",
        f"  subset  : {config.dataset.subset} batches (-1 = all)",
        "",
    ])
    with open(info_out, 'w') as f:
        f.write(dataset_info)
        f.write(neuron_summary + "\n")
        f.write(comparison_table)
    print(f"Saved prune info  -> {info_out}")

    ckpt_out = output_dir / f'checkpoint_t{threshold_str}_.pth'
    torch.save({
        'epoch': checkpoint_data.get('epoch', 0),
        'best_val_loss': pruned_val_loss,
        'model_state_dict': pruned_model.state_dict(),
        'optimizer_state_dict': None,
        'lr_scheduler_state_dict': None,
        'continue_from': str(checkpoint_path),
    }, ckpt_out)
    print(f"Saved pruned checkpoint -> {ckpt_out}")


if __name__ == '__main__':
    main()
