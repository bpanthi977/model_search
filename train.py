import math

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import datetime

from config import TrainConfig, OptimizerConfig
from dataset import Dataset
from model import MLP

def get_loss_fn(loss: str):
    if loss == 'mse':
        return nn.MSELoss(reduction='sum')
    elif loss == 'mae':
        return nn.L1Loss(reduction='sum')
    elif loss == 'smoothl1':
        return nn.SmoothL1Loss(reduction='sum', beta=1.0)
    else:
        assert false, f"Impossible value of loss function {loss}. Fix validate in TrainConfig."

    return loss_fn

def get_optimizer(config: OptimizerConfig, model: nn.Module):
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        assert false, f"Impossible value of optimizer {config.optimizer}. Fix validate in OptimizerConfig."

def train(model: MLP, dataset: Dataset, config: TrainConfig, callbacks):
    train_dataset = TensorDataset(torch.from_numpy(dataset.trainX), torch.from_numpy(dataset.trainY))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(dataset.validateX), torch.from_numpy(dataset.validateY))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    epoch_bar = tqdm(range(config.epoch), unit="epoch")
    device = model.get_device()

    optimizer = get_optimizer(config.optim, model)
    loss_fn = get_loss_fn(config.loss)

    for epoch in epoch_bar:

        model.train()
        total_loss = 0.0
        batch_bar = tqdm(train_dataloader, unit="batch", leave=False)
        start = datetime.now()
        for batch_X, batch_Y in batch_bar:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            Y_pred = model.forward(batch_X)
            loss = loss_fn(Y_pred, batch_Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mean_loss = loss.item() / batch_X.shape[0]
            batch_bar.set_postfix({f"{config.loss}": f"{mean_loss:.4f}"})

        train_time = (datetime.now() - start).microseconds / 1000.0
        mean_train_loss = total_loss / len(train_dataloader.dataset)

        model.eval()
        batch_bar = tqdm(val_dataloader, unit="batch", leave=False)
        total_loss = 0.0
        start = datetime.now()
        for batch_X, batch_Y in batch_bar:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            Y_pred = model.forward(batch_X)
            loss = loss_fn(Y_pred, batch_Y)

            total_loss += loss.item()
            mean_loss = loss.item() / batch_X.shape[0]
            batch_bar.set_postfix({f"{config.loss}": f"{mean_loss:.4f}"})

        val_time = (datetime.now() - start).microseconds / 1000.0
        mean_val_loss = total_loss / len(val_dataloader.dataset)

        epoch_bar.set_postfix({
            f"Train {config.loss}": f"{mean_train_loss:.4f}",
            f"Val {config.loss}": f"{mean_val_loss:.4f}"
        })

        for callback in callbacks:
            callback({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "train_time": train_time,
                "val_time": val_time
            })
