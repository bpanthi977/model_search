from pathlib import Path
import argparse
import csv

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def visualize_weights(model: torch.nn.Module, path: Path):
    count = 0
    for p in model.parameters():
        if len(p.shape) == 2:
            # Matrix
            plt.figure()
            sns.heatmap(p.detach().numpy())
            plt.savefig(path.joinpath(f"{count:03d}-A"))
        elif len(p.shape) == 1:
            # Vector
            plt.figure()
            sns.heatmap(p.reshape(-1, 1).detach().numpy())
            plt.savefig(path.joinpath(f"{count:03d}-b"))

        count += 1


def visualize_loss(log_dir: Path, fig_dir: Path, figsize=None):
    def read_loss(path):
        with open(path, "r") as f:
            losses = [(int(row[0]), float(row[1])) for row in csv.reader(f)]
        return pd.DataFrame(losses, columns=("epoch", "loss"))


    tl = read_loss(log_dir.joinpath("train_loss.csv"))
    tl["type"] = "Train"

    vl = read_loss(log_dir.joinpath("val_loss.csv"))
    vl["type"] = "Validation"

    df = pd.concat([tl, vl], ignore_index=True)

    fs = 8
    plt.rcParams.update({
        'font.size': fs,
        'axes.titlesize': fs,
        'axes.labelsize': fs,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'legend.fontsize': fs,
        'legend.title_fontsize': fs
    })

    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=df, x='epoch', y='loss', hue='type')
    ax.set_yscale('log')
    ax.set_title("Training and Validation Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    plt.legend(title="Loss Type")
    plt.tight_layout()

    plt.savefig(fig_dir.joinpath("loss_curve.svg"))
    plt.savefig(fig_dir.joinpath("loss_curve.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Visualizer",
        description="Visualize model weights and biases; Loss curves"
    )

    parser.add_argument("--log-dir",help="Logs dir from where to read loss file and model file.")
    parser.add_argument("--model", help="Model file (Default: model.pt in provided log-dir folder)")

    args = parser.parse_args()
    model_path = False
    log_path = False
    if args.log_dir:
        log_path = Path(args.log_dir)
        if not log_path.exists():
            print(f"Log dir {log_path} doesn't exist.")
            exit(1)

        model_path = log_path.joinpath("model.pt")
        if not model_path.exists():
            model_path = False


    if args.model:
        model_path = Path(args.model)

        if not model_path.exists():
            print(f"Model Path {model_path} doesn't exist.")
            exit(1)

    if log_path:
        fig_dir = log_path.joinpath("figs")
    elif model_path:
        fig_dir = model_path.parent.joinpath("figs")
    else:
        print("Provide at least one of model or log path. See --help.")
        exit(1)


    fig_dir.mkdir(parents=True, exist_ok=True)

    if log_path:
        visualize_loss(log_path, fig_dir)

    if model_path:
        model = torch.jit.load(model_path)
        visualize_weights(model, fig_dir)
        parameter_count = sum(param.numel() for param in model.parameters())
        print(f"parameter_count: {parameter_count}")

    exit(0)
