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
    hist_diff = torch.zeros(3 * batches, dtype=torch.double)
    hist_pred = torch.zeros(3 * batches, dtype=torch.double)
    hist_act = torch.zeros(3 * batches, dtype=torch.double)
    val_stat = MinMax()
    def set_hist(hist, i, stat):
        hist[3*i+0] = stat.min().item()
        hist[3*i+1] = stat.max().item()
        hist[3*i+2] = torch.median(stat).item()

    with torch.no_grad():
        for i, (batch_X, batch_Y) in enumerate(batch_bar):
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            Y_pred = model.forward(batch_X)
            loss = loss_fn(Y_pred, batch_Y)

            total_loss += loss.detach().item()

            diff = (Y_pred - batch_Y).detach()
            set_hist(hist_pred, i, Y_pred)
            set_hist(hist_act, i, batch_Y)
            set_hist(hist_diff, i, diff)
            val_stat.update(diff)

            batch_bar.set_postfix({
                f"ΔY": val_stat.current()
            })

    mean_val_loss = total_loss.item() / len(val_dataloader.dataset)
    print(f"Validation Loss = {mean_val_loss}")
    print(f"Min, Max        = {val_stat.agg()}")

    def draw_hist(hist, title: str, xlabel: str, file_name:str):
        plt.figure(figsize=(8,5))
        sns.histplot(hist.numpy(), color="blue", stat="count", alpha=0.6, bins=30)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        checkpoint_path.joinpath('figs/').mkdir(parents=True, exist_ok=True)
        fig_path = checkpoint_path.joinpath(f"figs/{file_name}.png")
        plt.savefig(fig_path)
        print("Histogram saved at:", fig_path)


    draw_hist(hist_diff, "Histogram of Error", "Y_pred - Y", "val_error_histogram")
    draw_hist(hist_pred, "Histogram of Prediction Values", "Y_pred", "y_pred_histogram")
    draw_hist(hist_act, "Histogram of Actual Values", "Y", "y_act_histogram")

    plt.figure(figsize=(8,5))
    plt.scatter(hist_act.numpy(), hist_diff.numpy(), s=9)
    plt.xlabel('Actual Values')
    plt.ylabel('Error (Y_pred - Y)')
    plt.title('Error vs Actual value')
    fig_path = checkpoint_path.joinpath("figs/relative_error.png")
    plt.savefig(fig_path)
    print("Scatter plot saved at: ", fig_path)

    return mean_val_loss

def create_model_from_checkpoint(config: Config, checkpoint_path: Path):
    checkpoint: Checkpoint = torch.load(checkpoint_path.joinpath('checkpoint.pth'))
    if not checkpoint:
        print(f"Invalid checkpoint directory {checkpoint_path}")
        exit(1)

    dataset = load_dataset(config.dataset)
    model = create_model(config.train, dataset)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    print(f"Saved best val loss: {checkpoint['best_val_loss']}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.cpu()
    model.double()
    model_script = torch.jit.script(model)
    model_path = checkpoint_path.joinpath("model.pt")
    model_script.save(model_path)
    print(f"Model saved to: {model_path}")
