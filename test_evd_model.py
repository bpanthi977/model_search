"""
EVD PyTorch Model Validation Script

Purpose:
This script tests a trained PyTorch neural network model against the original mathematical 
data-generating function for the EVD (CalcElemVolumeDerivative) kernel from LULESH. It directly 
compares the 24 spatial input features (x[0..7], y[0..7], z[0..7]) to the 24 output derivatives 
(dvdx[0..7], dvdy[0..7], dvdz[0..7]).

Usage:
    python test_evd_model.py --model path/to/model.pt [OPTIONS]

Arguments:
    --model     (Required) Path to the exported TorchScript (.pt) model.
    --dataset   (Optional) Path to .h5 or .csv file to sample a realistic baseline.
    --vary-in   (Optional) Space-separated list of input indices (0-23) to sweep and generate
                a 6x4 grid of 1D plots for. Example: `--vary-in 0 8 16`
    --out-dir   Directory to save the generated plots and the `cmd.sh` reproducibility script.
    --range     Min and max bounds for the sweeping features (default: -1.0 1.0)
    --steps     Number of points in the sweep (default: 100)

Visualizations Generated:
    1. Jacobian Heatmaps: 24x24 local sensitivity matrices (True vs Model vs Error).
    2. Global Max Error: 24x24 heatmap of maximum absolute error during full sweeps.
    3. 6x4 Output Grids: 1D sweep plots mapping 1 varied input against all 24 outputs.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def get_in_name(idx):
    if idx < 8: return f"x[{idx}]"
    elif idx < 16: return f"y[{idx-8}]"
    else: return f"z[{idx-16}]"

def get_out_name(idx):
    if idx < 8: return f"dvdx[{idx}]"
    elif idx < 16: return f"dvdy[{idx-8}]"
    else: return f"dvdz[{idx-16}]"

def volu_der_batch(x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5, z0, z1, z2, z3, z4, z5):
    """Vectorized version of VoluDer from lulesh.cc"""
    twelfth = 1.0 / 12.0
    dvdx = (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) + (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) - (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5)
    dvdy = - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) - (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) + (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5)
    dvdz = - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) - (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) + (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5)
    return dvdx * twelfth, dvdy * twelfth, dvdz * twelfth

def calc_elem_volume_derivative_batch(X):
    """
    Vectorized version of CalcElemVolumeDerivative from lulesh.cc.
    X is shape (N, 24).
    Returns Y of shape (N, 24).
    """
    N = X.shape[0]
    Y = np.zeros((N, 24), dtype=np.float64)
    x = X[:, 0:8]
    y = X[:, 8:16]
    z = X[:, 16:24]
    
    # Map nodes according to lulesh.cc CalcElemVolumeDerivative
    Y[:, 0], Y[:, 8], Y[:, 16] = volu_der_batch(x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,7], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,7], z[:,1], z[:,2], z[:,3], z[:,4], z[:,5], z[:,7])
    Y[:, 3], Y[:, 11], Y[:, 19] = volu_der_batch(x[:,0], x[:,1], x[:,2], x[:,7], x[:,4], x[:,6], y[:,0], y[:,1], y[:,2], y[:,7], y[:,4], y[:,6], z[:,0], z[:,1], z[:,2], z[:,7], z[:,4], z[:,6])
    Y[:, 2], Y[:, 10], Y[:, 18] = volu_der_batch(x[:,3], x[:,0], x[:,1], x[:,6], x[:,7], x[:,5], y[:,3], y[:,0], y[:,1], y[:,6], y[:,7], y[:,5], z[:,3], z[:,0], z[:,1], z[:,6], z[:,7], z[:,5])
    Y[:, 1], Y[:, 9], Y[:, 17] = volu_der_batch(x[:,2], x[:,3], x[:,0], x[:,5], x[:,6], x[:,4], y[:,2], y[:,3], y[:,0], y[:,5], y[:,6], y[:,4], z[:,2], z[:,3], z[:,0], z[:,5], z[:,6], z[:,4])
    Y[:, 4], Y[:, 12], Y[:, 20] = volu_der_batch(x[:,7], x[:,6], x[:,5], x[:,0], x[:,3], x[:,1], y[:,7], y[:,6], y[:,5], y[:,0], y[:,3], y[:,1], z[:,7], z[:,6], z[:,5], z[:,0], z[:,3], z[:,1])
    Y[:, 5], Y[:, 13], Y[:, 21] = volu_der_batch(x[:,4], x[:,7], x[:,6], x[:,1], x[:,0], x[:,2], y[:,4], y[:,7], y[:,6], y[:,1], y[:,0], y[:,2], z[:,4], z[:,7], z[:,6], z[:,1], z[:,0], z[:,2])
    Y[:, 6], Y[:, 14], Y[:, 22] = volu_der_batch(x[:,5], x[:,4], x[:,7], x[:,2], x[:,1], x[:,3], y[:,5], y[:,4], y[:,7], y[:,2], y[:,1], y[:,3], z[:,5], z[:,4], z[:,7], z[:,2], z[:,1], z[:,3])
    Y[:, 7], Y[:, 15], Y[:, 23] = volu_der_batch(x[:,6], x[:,5], x[:,4], x[:,3], x[:,2], x[:,0], y[:,6], y[:,5], y[:,4], y[:,3], y[:,2], y[:,0], z[:,6], z[:,5], z[:,4], z[:,3], z[:,2], z[:,0])
    
    return Y

def load_baseline(dataset_path):
    """Sample a random baseline input row of size 24."""
    if not dataset_path:
        print("No dataset provided. Using random baseline in [-1, 1].")
        return np.random.rand(24) * 2 - 1.0
    
    p = Path(dataset_path)
    if not p.exists():
        print(f"Dataset {dataset_path} not found. Using random baseline.")
        return np.random.rand(24) * 2 - 1.0

    if p.suffix == '.h5':
        import h5py
        with h5py.File(p, 'r') as f:
            if 'EVD_train' in f:
                data = f['EVD_train']['input']
            elif 'EVD' in f:
                data = f['EVD']['input']
            else:
                raise ValueError("Could not find 'EVD_train' or 'EVD' in H5 file.")
            idx = np.random.randint(0, data.shape[0])
            row = np.array(data[idx]).flatten()
            print(f"Sampled baseline from HDF5 index {idx}.")
            return row[:24]
    elif p.suffix == '.csv':
        data = np.loadtxt(p, delimiter=',')
        idx = np.random.randint(0, data.shape[0])
        print(f"Sampled baseline from CSV index {idx}.")
        return data[idx][:24]
    else:
        raise ValueError("Unsupported dataset format. Use .h5 or .csv")

def evaluate_model(model, X_np):
    """Evaluate PyTorch model, handling casting to double precision."""
    with torch.no_grad():
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        X_t = torch.tensor(X_np, dtype=torch.float64).to(device)
        Y_pred = model(X_t).cpu().numpy()
    return Y_pred

def main():
    parser = argparse.ArgumentParser(description="Test PyTorch model against true EVD function.")
    parser.add_argument('--model', required=True, help="Path to the .pt TorchScript model.")
    parser.add_argument('--dataset', default=None, help="Path to .h5 or .csv file to sample baseline.")
    parser.add_argument('--vary-in', nargs='*', type=int, default=[], help="List of input indices (0-23) to vary and plot.")
    parser.add_argument('--out-dir', default='evd_test_results', help="Output directory for plots.")
    parser.add_argument('--range', nargs=2, type=float, default=[-1.0, 1.0], help="Min and max for sweep.")
    parser.add_argument('--steps', type=int, default=100, help="Number of steps in sweep.")
    
    args = parser.parse_args()
    
    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the command to cmd.sh
    cmd_script = out_dir / "cmd.sh"
    with open(cmd_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("python " + " ".join(sys.argv) + "\n")
    os.chmod(cmd_script, 0o755)
    
    print(f"Loading TorchScript model from {args.model}...")
    model = torch.jit.load(args.model)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    baseline_X = load_baseline(args.dataset)
    
    # Labels for axes
    in_labels = [get_in_name(i) for i in range(24)]
    out_labels = [get_out_name(i) for i in range(24)]
    
    # =========================================================================
    # Idea 1: Local Jacobian / Sensitivity Heatmaps
    # =========================================================================
    print("Generating Jacobian heatmaps...")
    h = 1e-5
    X_plus = np.tile(baseline_X, (24, 1))
    X_minus = np.tile(baseline_X, (24, 1))
    for i in range(24):
        X_plus[i, i] += h
        X_minus[i, i] -= h
        
    Y_true_plus = calc_elem_volume_derivative_batch(X_plus)
    Y_true_minus = calc_elem_volume_derivative_batch(X_minus)
    J_true = ((Y_true_plus - Y_true_minus) / (2 * h)).T  # shape (outputs, inputs)
    
    Y_model_plus = evaluate_model(model, X_plus)
    Y_model_minus = evaluate_model(model, X_minus)
    J_model = ((Y_model_plus - Y_model_minus) / (2 * h)).T
    
    J_diff = np.abs(J_true - J_model)
    
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    
    sns.heatmap(J_true, ax=axes[0], cmap="coolwarm", center=0, xticklabels=in_labels, yticklabels=out_labels)
    axes[0].set_title("True Function Jacobian")
    axes[0].set_xlabel("Input Feature")
    axes[0].set_ylabel("Output Feature")
    
    sns.heatmap(J_model, ax=axes[1], cmap="coolwarm", center=0, xticklabels=in_labels, yticklabels=out_labels)
    axes[1].set_title("Model Jacobian")
    axes[1].set_xlabel("Input Feature")
    
    sns.heatmap(J_diff, ax=axes[2], cmap="Reds", xticklabels=in_labels, yticklabels=out_labels)
    axes[2].set_title("Absolute Difference")
    axes[2].set_xlabel("Input Feature")
    
    plt.tight_layout()
    plt.savefig(out_dir / "01_jacobian_heatmaps.png")
    plt.close()
    
    # =========================================================================
    # Idea 2: Global Sweep Max Error Matrix
    # =========================================================================
    print("Generating Sweep Max Error Matrix...")
    sweep_vals = np.linspace(args.range[0], args.range[1], args.steps)
    sweep_error_matrix = np.zeros((24, 24))  # (outputs, inputs)
    
    for in_idx in range(24):
        X_sweep = np.tile(baseline_X, (args.steps, 1))
        X_sweep[:, in_idx] = sweep_vals
        
        Y_true = calc_elem_volume_derivative_batch(X_sweep)
        Y_model = evaluate_model(model, X_sweep)
        
        abs_err = np.abs(Y_true - Y_model)
        max_err = np.max(abs_err, axis=0)  # shape (24,)
        sweep_error_matrix[:, in_idx] = max_err
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(sweep_error_matrix, cmap="Reds", xticklabels=in_labels, yticklabels=out_labels)
    plt.title("Max Absolute Error over Sweeps\n(Row=Output, Col=Varying Input)")
    plt.xlabel("Varying Input Feature")
    plt.ylabel("Output Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "02_sweep_max_error_matrix.png")
    plt.close()
    
    # =========================================================================
    # Custom 1D Line Plots (6x4 Grid per varied input)
    # =========================================================================
    for in_idx in args.vary_in:
        in_name = get_in_name(in_idx)
        print(f"Generating 24-plot grid varying {in_name}...")
        
        X_sweep = np.tile(baseline_X, (args.steps, 1))
        X_sweep[:, in_idx] = sweep_vals
        
        Y_true = calc_elem_volume_derivative_batch(X_sweep)
        Y_model = evaluate_model(model, X_sweep)
        
        fig, axes = plt.subplots(6, 4, figsize=(16, 12), sharex=True)
        fig.suptitle(f"Sweeping Input {in_name} (from {args.range[0]} to {args.range[1]})", fontsize=16)
        
        for out_idx in range(24):
            row = out_idx // 4
            col = out_idx % 4
            ax = axes[row, col]
            
            ax.plot(sweep_vals, Y_true[:, out_idx], label="True Function", color='blue', linewidth=2)
            ax.plot(sweep_vals, Y_model[:, out_idx], label="Model Prediction", color='red', linestyle='--', linewidth=2)
            
            ax.set_title(get_out_name(out_idx), fontsize=10)
            ax.grid(True, alpha=0.5)
            
            if row == 5:
                ax.set_xlabel(f"Value of {in_name}", fontsize=9)
                
        # Add a single legend for the whole figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.4, 0.94), ncol=2)
        
        # Add a text box with the constant features in a 3-column Node layout
        textstr = "Constant Features (Baseline):\n\n"
        lines = []
        for node in range(8):
            x_idx, y_idx, z_idx = node, node + 8, node + 16
            
            x_str = f"x[{node}]={baseline_X[x_idx]:.4f}" if x_idx != in_idx else f"x[{node}]=[Varied]"
            y_str = f"y[{node}]={baseline_X[y_idx]:.4f}" if y_idx != in_idx else f"y[{node}]=[Varied]"
            z_str = f"z[{node}]={baseline_X[z_idx]:.4f}" if z_idx != in_idx else f"z[{node}]=[Varied]"
            
            lines.append(f"Node {node}:\n  {x_str:<16}\n  {y_str:<16}\n  {z_str:<16}")
            
        # Re-format to strictly 3 columns side-by-side
        lines_compact = []
        for node in range(8):
            x_idx, y_idx, z_idx = node, node + 8, node + 16
            x_str = f"x[{node}]={baseline_X[x_idx]:.4f}" if x_idx != in_idx else f"x[{node}]=[Varied]"
            y_str = f"y[{node}]={baseline_X[y_idx]:.4f}" if y_idx != in_idx else f"y[{node}]=[Varied]"
            z_str = f"z[{node}]={baseline_X[z_idx]:.4f}" if z_idx != in_idx else f"z[{node}]=[Varied]"
            lines_compact.append(f"Node {node:d}: {x_str:<15} {y_str:<15} {z_str:<15}")
            
        textstr += "\n".join(lines_compact)
        
        # Place text box to the right of the plot grid
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        fig.text(0.77, 0.5, textstr, fontsize=10, family='monospace', 
                 verticalalignment='center', bbox=props)
        
        # Adjust layout to make room for the text box on the right
        plt.subplots_adjust(right=0.74, top=0.88, hspace=0.4, wspace=0.3)
        
        safe_name = in_name.replace('[', '').replace(']', '')
        plt.savefig(out_dir / f"03_sweep_vary_in_{safe_name}.png", bbox_inches='tight')
        plt.close()

    print(f"All done! Results saved to '{out_dir}/'.")

if __name__ == "__main__":
    main()
