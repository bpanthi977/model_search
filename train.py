"""Train a model from given config and save logs."""

import csv
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pyrallis

from config import Config, TrainConfig, OptimizerConfig
from dataset import Dataset, load_dataset
from model import MLP, create_model
from visualize import visualize_weights, visualize_loss

class MinMax():
    def __init__(self):
        self.min = None
        self.max = None
        self.cur_min = 0.0
        self.cur_max = 0.0

    def update(self, tensor: torch.Tensor):
        self.cur_min = tensor.min().detach().item()
        self.cur_max = tensor.max().detach().item()

        if self.min:
            self.min = min(self.min, self.cur_min)
        else:
            self.min = self.cur_min

        if self.max:
            self.max = max(self.max, self.cur_max)
        else:
            self.max = self.cur_max

    def current(self):
        return f"({self.cur_min:.4f},{self.cur_max:.4f})"

    def agg(self):
        return f"({self.min:.4f},{self.max:.4f})"


def get_loss_fn(loss: str):
    """Create loss function."""
    if loss == 'mse':
        return nn.MSELoss(reduction='sum')
    elif loss == 'mae':
        return nn.L1Loss(reduction='sum')
    elif loss == 'smoothl1':
        return nn.SmoothL1Loss(reduction='sum', beta=1.0)

    assert False, f"Impossible value of loss function {loss}. Fix validate in TrainConfig."

def get_optimizer(config: OptimizerConfig, model: nn.Module):
    """Create optimizer for given config."""
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        assert False, f"Impossible value of optimizer {config.optimizer}. Fix validate in OptimizerConfig."

def train(model: MLP, dataset: Dataset, config: TrainConfig, callbacks):
    """
    Train the model with optimizer, loss function and other things as per config.

    Calls each function in `callbacks` with a dict of epoch, {train, val}_{loss, time}.
    """
    train_dataset = TensorDataset(dataset.trainX, dataset.trainY)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = TensorDataset(dataset.validateX, dataset.validateY)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    epoch_bar = tqdm(range(config.epoch), unit="epoch")
    device = model.get_device()

    optimizer = get_optimizer(config.optim, model)
    loss_fn = get_loss_fn(config.loss)

    print(model)

    for epoch in epoch_bar:

        model.train()
        total_loss = torch.tensor(0.0).to(device)
        batch_bar = tqdm(train_dataloader, unit="batch", leave=False)
        torch.cuda.synchronize()
        start = datetime.now()
        train_y = MinMax()
        for batch_X, batch_Y in batch_bar:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            if config.model.normalize:
                Y_pred = model.model(model.normalize(batch_X, model.normalizeX))
                batch_Y = model.normalize(batch_Y, model.normalizeY)
            else:
                Y_pred = model(batch_X);

            loss = loss_fn(Y_pred, batch_Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

            train_y.update(Y_pred - batch_Y)
            batch_bar.set_postfix({
                "ΔY": train_y.current()
            })


        torch.cuda.synchronize()
        train_time = (datetime.now() - start).microseconds / 1000.0
        mean_train_loss = total_loss.item() / len(train_dataloader.dataset)

        model.eval()
        batch_bar = tqdm(val_dataloader, unit="batch", leave=False)
        total_loss = torch.tensor(0.0).to(device)
        start = datetime.now()

        val_y = MinMax()
        with torch.no_grad():
            for batch_X, batch_Y in batch_bar:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                Y_pred = model.forward(batch_X)
                loss = loss_fn(Y_pred, batch_Y)

                total_loss += loss.detach()

                val_y.update(Y_pred - batch_Y)
                batch_bar.set_postfix({
                    f"ΔY": val_y.current()
                })

            torch.cuda.synchronize()
            val_time = (datetime.now() - start).microseconds / 1000.0
            mean_val_loss = total_loss.item() / len(val_dataloader.dataset)

        epoch_bar.set_postfix({
            f"Train {config.loss}": f"{mean_train_loss:.4f}",
            f"Val {config.loss}": f"{mean_val_loss:.4f}",
            f"ΔY": train_y.agg()
        })

        for callback in callbacks:
            callback({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "train_time": train_time,
                "val_time": val_time
            })

def format_duration(d):
    """Format to h:m:s format."""
    ts = d.total_seconds()
    h = int(ts // (60*60))
    m = int((ts - h * 60 * 60) // 60)
    s = int(ts - h * 60 * 60 - m * 60)
    return f"{h}:{m}:{s}"

def train_log(config: Config, trial_id: int | str, callbacks, dataset = Optional[Dataset]) -> float:
    """Start training and save config, model, loss curves to logs_dir/study_name/trail_id directory."""
    # Save config and loss
    start = datetime.now()
    run_dir = config.logs_dir.joinpath(config.study_name).joinpath(f"{trial_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    final_val_loss = None
    with open(run_dir.joinpath("config.yaml"), "w") as f:
        f.write(pyrallis.dump(config))
        print(config)

    with (
            open(run_dir.joinpath("train_loss.csv"), "w", newline='') as f_train,
            open(run_dir.joinpath("val_loss.csv"), "w", newline='') as f_val
    ):
        w_train = csv.writer(f_train)
        w_val = csv.writer(f_val)

        def callback(info):
            nonlocal final_val_loss

            epoch = info["epoch"]
            final_val_loss = info["val_loss"]
            w_train.writerow([epoch+1, info["train_loss"], info["train_time"]])
            w_val.writerow([epoch+1, final_val_loss, info["val_time"]])
            f_train.flush()
            f_val.flush()

        # Load data, train and evaluate
        if not dataset:
            dataset = load_dataset(config.dataset)

        model = create_model(config.train, dataset)
        with open(run_dir.joinpath("model_shape"), "w") as f:
            f.write(str(model))

        try:
            train(model, dataset, config.train, [callback, *callbacks])
        except KeyboardInterrupt:
            pass

    # Save model
    model_script = torch.jit.script(model.to(torch.device('cpu')).double())
    model_script.save(run_dir.joinpath("model.pt"))

    with open(run_dir.joinpath("info.csv"), "w", newline='') as f:
        csv.writer(f).writerows(
            [["start_time", start.strftime("%Y%m%d-%H:%M:%S")],
             ["end_time", datetime.now().strftime("%Y%m%d-%H:%M:%S")],
             ["total_time", format_duration(datetime.now() - start)],
             ["val_loss", final_val_loss],
             ["loss", config.train.loss],
             ["training_size", dataset.trainX.shape[0]],
             ["validation_size", dataset.validateX.shape[0]]]
        )

    # Create visualizations
    fig_dir = run_dir.joinpath("figs")
    fig_dir.mkdir(parents=True, exist_ok=True)
    visualize_weights(model, fig_dir)
    visualize_loss(run_dir, fig_dir)

    # Return the evaluation metric
    return final_val_loss
