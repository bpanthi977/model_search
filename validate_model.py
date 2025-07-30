from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import load_dataset
from train import Checkpoint, MinMax, create_model, get_loss_fn
from config import Config

def validate(config: Config, checkpoint_path: Path):
    checkpoint: Checkpoint = torch.load(checkpoint_path.joinpath('checkpoint.pth'))
    if not checkpoint:
        print(f"Invalid checkpoint directory {checkpoint_path}")
        exit(1)

    dataset = load_dataset(config.dataset)
    model = create_model(config.train, dataset)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    print(f"Saved best val loss: {checkpoint['best_val_loss']}")

    val_dataset = TensorDataset(dataset.validateX, dataset.validateY)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    device = model.get_device()
    loss_fn = get_loss_fn(config.train.loss)

    batch_bar = tqdm(val_dataloader)
    total_loss = torch.tensor(0.0)
    batches = len(batch_bar)
    hist = torch.zeros(3 * batches, dtype=torch.double)
    val_stat = MinMax()
    with torch.no_grad():
        for i, (batch_X, batch_Y) in enumerate(batch_bar):
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            Y_pred = model.forward(batch_X)
            loss = loss_fn(Y_pred, batch_Y)

            total_loss += loss.detach().item()

            diff = (Y_pred - batch_Y).detach()
            hist[3*i+0] = diff.min().item()
            hist[3*i+1] = diff.max().item()
            hist[3*i+2] = torch.median(diff).item()
            val_stat.update(diff)

            batch_bar.set_postfix({
                f"ΔY": val_stat.current()
            })

    mean_val_loss = total_loss.item() / len(val_dataloader.dataset)
    print(f"Validation Loss = {mean_val_loss}")
    print(f"Min, Max        = {val_stat.agg()}")

    plt.figure(figsize=(8,5))
    sns.histplot(hist.numpy(), color="blue", label="Error", stat="count", alpha=0.6, bins=30)
    plt.title("Histogram of Two Datasets")
    plt.xlabel("Y_pred - Y")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    checkpoint_path.joinpath('figs/').mkdir(parents=True, exist_ok=True)
    fig_path = checkpoint_path.joinpath("figs/val_error_histogram.png")
    plt.savefig(fig_path)
    print("Histogram saved at:", fig_path)
    return mean_val_loss
