import argparse
import os
import sys
import shutil
from pathlib import Path
import copy
import pyrallis
from datetime import datetime
import random
import string
from typing import Optional, List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import math

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

from config import Config, TrainConfig, OptimizerConfig, parse_lr_scheduler
from dataset import Dataset, load_dataset
from model import create_model
from train import get_optimizer, get_loss_fn

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) # Not all environments have cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_lr_tune(config: Config, dataset: Optional[Dataset] = None):
    # Setup directories

    seed_bytes = os.urandom(8)
    rng = random.Random(int.from_bytes(seed_bytes, byteorder='big'))
    random_suffix = ''.join([rng.choice(string.ascii_letters) for i in range(4)])

    trial_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + random_suffix
    run_dir = Path("logs_lr_tune").joinpath(config.study_name).joinpath(f"{trial_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running LR Range Test. Logs will be saved to: {run_dir}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(run_dir))

    # Save config
    with open(run_dir.joinpath("config.yaml"), "w") as f:
        f.write(pyrallis.dump(config))

    # Load dataset if not provided
    if not dataset:
        print("Loading dataset...")
        dataset = load_dataset(config.dataset)

    # Create model
    print("Creating model...")
    model = create_model(config.train, dataset)
    device = model.get_device()
    print(f"Model created on device: {device}")

    # Setup Optimizer with tiny initial LR
    initial_lr = 1e-7
    optimizer = get_optimizer(config.train.optim, model)

    # Manually set initial LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

    loss_fn = get_loss_fn(config.train.loss)

    # Data Loader
    train_dataset = TensorDataset(dataset.trainX, dataset.trainY)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)

    # LR Range Test Parameters
    lrs = []
    losses = []

    # Target range: 1e-7 to 10
    final_lr = 10.0
    num_steps = 100 # Target steps

    # Calculate multiplier
    lr_multiplier = (final_lr / initial_lr) ** (1 / num_steps)

    print(f"LR Scheme: Start={initial_lr}, End={final_lr}, Steps={num_steps}, Multiplier={lr_multiplier:.5f}")

    # Log hyperparameters
    hparams = {
        "study_name": config.study_name,
        "dataset": config.dataset.db_file,
        "model_init": config.train.model.init,
        "model_activation": config.train.model.activation,
        "normalize": config.train.model.normalize,
        "optimizer": config.train.optim.optimizer,
        "loss": config.train.loss,
        "weight_decay": config.train.optim.weight_decay,
        "batch_size": config.train.batch_size,
        "initial_lr": initial_lr,
        "final_lr": final_lr,
        "num_steps": num_steps
    }
    # Add hidden layers configuration
    if config.train.model.hidden_layers:
         hparams["hidden_layers"] = str(config.train.model.hidden_layers)

    writer.add_hparams(hparams, metric_dict={})

    model.train()

    step_count = 0
    best_loss = float('inf')

    print("Starting training loop for LR tuning...")

    iterator = iter(train_dataloader)

    pbar = tqdm(total=num_steps)

    stop_training = False

    while not stop_training:
        try:
            batch_X, batch_Y = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)
            batch_X, batch_Y = next(iterator)

        step_count += 1

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()

        # Forward pass (Normalized)
        # Note: model.model is the underlying nn.Sequential which expects normalized inputs and returns normalized outputs
        Y_pred = model.model(model.normalize(batch_X, model.normalizeX))
        batch_Y_norm = model.normalize(batch_Y, model.normalizeY)

        loss = loss_fn(Y_pred, batch_Y_norm) / batch_X.shape[0]

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        # Log to TensorBoard
        writer.add_scalar('Train/Loss', current_loss, step_count)
        writer.add_scalar('Train/LR', current_lr, step_count)

        lrs.append(current_lr)
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss

        # Stop if loss explodes
        # User requested a very large threshold
        if current_loss > 4 * losses[0]:
            print(f"Loss exploded at step {step_count} (LR: {current_lr:.6f}, Loss: {current_loss:.4f}, Best: {best_loss:.4f}). Stopping.")
            stop_training = True

        if step_count >= num_steps:
             print(f"Reached max steps {num_steps}. Stopping.")
             stop_training = True

        if not stop_training:
            # Increase LR
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_multiplier

        pbar.update(1)
        pbar.set_postfix({"lr": f"{current_lr:.6f}", "loss": f"{current_loss:.4f}"})

    pbar.close()

    # Save Logs
    csv_path = run_dir.joinpath("lr_log.csv")
    with open(csv_path, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["lr", "loss"])
        csv_writer.writerows(zip(lrs, losses))

    print(f"Saved logs to {csv_path}")

    # Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.grid(True, which="both", ls="-", alpha=0.5)

        for ext in ['png', 'pdf', 'svg']:
            plot_path = run_dir.joinpath(f'lr_plot.{ext}')
            plt.savefig(plot_path)
        print(f"Saved plot to {plot_path} (and others)")
    except Exception as e:
        print(f"Failed to create plot: {e}")

    # Close the writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Range Test")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config file")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        exit(1)

    try:
        config = pyrallis.parse(config_class=Config, config_path=args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        exit(1)

    set_all_seeds(42)
    run_lr_tune(config)
