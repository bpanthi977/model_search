import os
import sys
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def read_array(f):
    n_dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    shape = np.fromfile(f, dtype=np.int32, count=n_dims)
    data = np.fromfile(f, dtype=np.double, count=np.prod(shape)).reshape(shape)
    return data

def read_energy_file(path):
    dataset = []
    with open(path, "rb") as f:
        i = 0
        stop = False
        while not stop:
            try:
                dataset.append(read_array(f))
            except:
                stop = True
            i += 1

    return np.array(dataset)

def create_heatmap(original, model, t=0, x=0):
    e1 = original[t][x]
    e1 = np.log10(np.clip(e1, a_min=1e-3, a_max=1e8))

    e2 = model[t][x]
    e2 = np.log10(np.clip(e2, a_min=1e-3, a_max=1e8))

    ediff = model[t][x] - original[t][x]
    ediff = np.log10(np.clip(np.abs(ediff), a_min=1e-3, a_max=1e8))

    vmin = min(np.min(e1), np.min(e2))
    vmax = max(np.max(e1), np.max(e2))

    cmap = 'viridis'

    fig = plt.figure(figsize=(8, 2.5))
    gs = fig.add_gridspec(1, 4, wspace=0.15, width_ratios=[1, 1, 1, 0.05])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    fs=8
    sns.heatmap(e1, cmap=cmap,ax=axes[0], vmin=vmin, vmax=vmax, linewidths=0.0, rasterized=True, cbar=False)
    axes[0].set_title(f'Original Energy Heatmap', fontsize=fs)
    axes[0].set_xlabel(f'Y-coordinate', fontsize=fs)
    axes[0].set_ylabel(f'Z-coordinate', fontsize=fs)
    sns.heatmap(e2, cmap=cmap, ax=axes[1], vmin=vmin, vmax=vmax, linewidths=0.0, rasterized=True, cbar=False)
    axes[1].set_title(f'Energy heatmap from Model', fontsize=fs)
    axes[1].set_xlabel(f'Y-coordinate', fontsize=fs)
    axes[1].set_ylabel(f'Z-coordinate', fontsize=fs)
    sns.heatmap(ediff, cmap=cmap, ax=axes[2], vmin=vmin, vmax=vmax, linewidths=0.0, rasterized=True, cbar_ax=cbar_ax)
    axes[2].set_title(f'Absolute Difference (log scale)', fontsize=fs)
    axes[2].set_xlabel(f'Y-coordinate', fontsize=fs)
    axes[2].set_ylabel(f'Z-coordinate', fontsize=fs)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False, labelsize=fs)

        if ax.collections:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=fs)

    plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Energy Diff",
        description="Compute difference in energy output."
    )

    parser.add_argument("--original",help="Original Energy file.")
    parser.add_argument("--model", help="Energy file when using model.")
    parser.add_argument("--visualize", action='store_true', help="Visualize the energy difference in a heatmap.")

    args = parser.parse_args()

    if not args.model:
        print(f"Energy file when using model is not provided. Use --model")
        exit(1)

    model_file = Path(args.model)

    if not model_file.exists():
        print(f"File --mode {model_file} doesn't exits.")
        exit(1)


    if not args.original:
        original_file = model_file.parent.joinpath("Energy_Original.bin")
    else:
        original_file = Path(args.original)

    if not original_file.exists():
        print(f"File --original {original_file} doesn't exist.")
        exit(1)

    model = read_energy_file(model_file)
    original = read_energy_file(original_file)

    print(f"Model Energy Shape   : {model.shape   }, file: {model_file}")
    print(f"Original Energy Shape: {original.shape}, file: {original_file}")

    total_nodes = model.shape[1] * model.shape[2] * model.shape[3]
    print(f"Total Energy (Original): {np.sum(original[-1])}")
    print(f"Avg Energy (Original): {np.sum(original[-1]) / total_nodes}")

    abs_diff = np.sum(np.abs(model[-1] - original[-1])).item()
    print(f"Total ABS Diff: {abs_diff}")
    print(f"Avg ABS Diff: {abs_diff / total_nodes}")

    sq_diff = np.sum((model[-1] - original[-1]) ** 2).item()
    print(f"Total Squared Diff: {sq_diff}")
    print(f"Avg Squared Diff: {sq_diff / total_nodes}")

    if args.visualize:
        t = min(original.shape[0], model.shape[0])-1
        create_heatmap(original, model, t=t, x=0)
        file_name = f"{model_file.stem}-{t}"
        plt.savefig(model_file.parent.joinpath(f"{file_name}.svg"), bbox_inches='tight')
        plt.savefig(model_file.parent.joinpath(f"{file_name}.pdf"), bbox_inches='tight')
