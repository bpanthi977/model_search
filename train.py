"""Train a model from given config and save logs."""

import csv
from datetime import datetime
from typing import Any, Optional, TypedDict
from pathlib import Path
import typing

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pyrallis

from config import Config, TrainConfig, OptimizerConfig, parse_lr_scheduler
from dataset import Dataset, load_dataset
from model import MLP, create_model, MULT0, MULT1
from visualize import visualize_weights, visualize_loss, visualize_model
from log_gpu_utilization import log_gpu_utilization

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

def collect_params(model: nn.Module):
    usual_params = []
    nalu_params = []

    for module in model.modules():
        if module is model:
            pass
        elif isinstance(module, MULT0) or isinstance(module, MULT1):
            nalu_params.extend(module.parameters(recurse=False))
        else:
            usual_params.extend(module.parameters(recurse=False))

    return usual_params, nalu_params

def get_optimizer(config: OptimizerConfig, model: nn.Module):
    """Create optimizer for given config."""
    lr: float = parse_lr_scheduler(config.lr)[1]
    nalu_lr = config.nalu_lr or lr
    usual_params, nalu_params = collect_params(model)
    param_groups = [
            {'params': usual_params}
    ]
    if len(nalu_params) > 0:
        param_groups.append({'params': nalu_params, 'lr': nalu_lr, 'weight_decay': 0.0})

    if config.optimizer == 'adamw':
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'adagrad':
        return torch.optim.Adagrad(param_groups, lr=lr, weight_decay=config.weight_decay)
    else:
        assert False, f"Impossible value of optimizer {config.optimizer}. Fix validate in OptimizerConfig."

def get_lr_scheduler(lr: str, optimizer: torch.optim.Optimizer, total_epochs: int, last_epoch: int = -1):
    args = parse_lr_scheduler(lr)
    lr_type = args[0]
    start = args[1]
    if lr_type == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, 1, total_epochs, last_epoch)
    elif lr_type == 'linear':
        end = args[2]
        def lambda_lr(epoch):
            if epoch >= total_epochs:
                return end / start
            else:
                return 1 + (end - start) / start * epoch / (total_epochs - 1)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch)
    else:
        raise Exception(f"[BUG] LR Schedule not supported '{lr}'")

class Env:
    epoch: Optional[int] = None
    best_val_loss: float = float('+inf')
    best_max_l1: float = float('+inf')
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(self):
        pass

class Checkpoint(TypedDict):
    epoch: int
    best_val_loss: float
    model_state_dict: dict
    optimizer_state_dict: dict
    lr_scheduler_state_dict: dict

def train(dataset: Dataset, config: TrainConfig, callbacks, env: Env, checkpoint: Optional[Checkpoint]):
    """
    Train the model with optimizer, loss function and other things as per config.

    Calls each function in `callbacks` with a dict of epoch, {train, val}_{loss, time}.
    """
    train_dataset = TensorDataset(dataset.trainX, dataset.trainY)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = TensorDataset(dataset.validateX, dataset.validateY)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    last_epoch = -1

    model = create_model(config, dataset)
    device = model.get_device()
    optimizer = get_optimizer(config.optim, model)
    loss_fn = get_loss_fn(config.loss)
    lr_scheduler = get_lr_scheduler(config.optim.lr, optimizer, config.epoch, last_epoch)

    env.model = model
    env.optimizer = optimizer
    env.lr_scheduler = lr_scheduler

    if checkpoint:
        last_epoch: int = checkpoint['epoch']
        env.best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    epoch_bar = tqdm(range(last_epoch + 1, config.epoch), unit="epoch")
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
            Y_pred = model.model(model.normalize(batch_X, model.normalizeX))
            batch_Y = model.normalize(batch_Y, model.normalizeY)

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

        val_l1 = MinMax()
        with torch.no_grad():
            for batch_X, batch_Y in batch_bar:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                Y_pred = model.forward(batch_X)
                loss = loss_fn(Y_pred, batch_Y)

                total_loss += loss.detach()

                val_l1.update(Y_pred - batch_Y)
                batch_bar.set_postfix({
                    f"ΔY": val_l1.current()
                })

            torch.cuda.synchronize()
            val_time = (datetime.now() - start).microseconds / 1000.0
            mean_val_loss = total_loss.item() / len(val_dataloader.dataset)

        epoch_bar.set_postfix({
            f"Train {config.loss}": f"{mean_train_loss:.4f}",
            f"Val {config.loss}": f"{mean_val_loss:.4f}",
            f"ΔY": train_y.agg(),
            f"lr": [pg['lr'] for pg in optimizer.param_groups]
        })

        max_l1 = max(abs(val_l1.max), abs(val_l1.min)) or float('+inf')
        env.epoch = epoch
        env.best_val_loss = min(env.best_val_loss, mean_val_loss)
        env.best_max_l1 = min(env.best_max_l1, max_l1)
        for callback in callbacks:
            callback({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "max_l1": max_l1,
                "train_time": train_time,
                "val_time": val_time,
                "all_lr": [pg['lr'] for pg in optimizer.param_groups]
            })

        lr_scheduler.step()

def format_duration(d):
    """Format to h:m:s format."""
    ts = d.total_seconds()
    h = int(ts // (60*60))
    m = int((ts - h * 60 * 60) // 60)
    s = int(ts - h * 60 * 60 - m * 60)
    return f"{h}:{m}:{s}"

def save_checkpoint(env: Env, checkpoint_file: Path):
    if env.epoch:
        checkpoint: Checkpoint = {
            'epoch': env.epoch,
            'best_val_loss': env.best_val_loss,
            'model_state_dict': env.model.state_dict(),
            'optimizer_state_dict': env.optimizer.state_dict(),
            'lr_scheduler_state_dict': env.lr_scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_file)

def train_log(config: Config, trial_id: int | str, callbacks, dataset = Optional[Dataset]) -> float:
    """Start training and save config, model, loss curves to logs_dir/study_name/trail_id directory."""
    # Save config and loss
    start = datetime.now()
    run_dir = config.logs_dir.joinpath(config.study_name).joinpath(f"{trial_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    last_info: dict = {}
    env = Env()

    with open(run_dir.joinpath("config.yaml"), "w") as f:
        f.write(pyrallis.dump(config))
        print(config)

    with (
            open(run_dir.joinpath("train_loss.csv"), "a", newline='') as f_train,
            open(run_dir.joinpath("val_loss.csv"), "a", newline='') as f_val
    ):
        w_train = csv.writer(f_train)
        w_val = csv.writer(f_val)

        def callback(info):
            nonlocal last_info
            last_info = info

            epoch = info["epoch"]
            val_loss = info["val_loss"]
            max_l1 = info['max_l1']
            if config.train.evaluation_metric == 'val_loss':
                if val_loss == env.best_val_loss:
                    save_checkpoint(env, run_dir.joinpath("checkpoint_best.pth"))
            elif config.train.evaluation_metric == 'max_l1':
                if max_l1 == env.best_max_l1:
                    save_checkpoint(env, run_dir.joinpath("checkpoint_best.pth"))
            else:
                raise ValueError(f"[BUG] invalid evaluation metric {config.train.evaluation_metric}")
            w_train.writerow([epoch+1, info["train_loss"], info["train_time"], info["all_lr"]])
            w_val.writerow([epoch+1, val_loss, info["val_time"], max_l1])
            f_train.flush()
            f_val.flush()

        # Start GPU utilization logging utility in the background
        stop_gpu_logging = False
        def stop_fn():
            return stop_gpu_logging

        # Load data, train and evaluate
        if not dataset:
            dataset = load_dataset(config.dataset)

        # Restore checkpoint
        checkpoint_path = run_dir.joinpath('checkpoint.pth')
        checkpoint: Optional[Checkpoint] = None
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)

        try:
            log_gpu_utilization(interval=10, log_file=run_dir.joinpath('gpu_utilization.csv'), stop_flag=stop_fn, new_thread=True)
            train(dataset, config.train, [callback, *callbacks], env, checkpoint)
        except KeyboardInterrupt:
            pass
        finally:
            stop_gpu_logging = True

    # Save model
    model = env.model
    try:
        model_script = torch.jit.script(model.to(torch.device('cpu')).double())
        model_script.save(run_dir.joinpath("model.pt"))
    except Exception as e:
        print(e)
    save_checkpoint(env, run_dir.joinpath('checkpoint.pth'))
    parameter_count = sum(param.numel() for param in model.parameters())

    with open(run_dir.joinpath("info.csv"), "w", newline='') as f:
        csv.writer(f).writerows(
            [["start_time", start.strftime("%Y%m%d-%H:%M:%S")],
             ["end_time", datetime.now().strftime("%Y%m%d-%H:%M:%S")],
             ["total_time", format_duration(datetime.now() - start)],
             ["val_loss", last_info['val_loss']],
             ["max_l1", last_info['max_l1']],
             ["best_val_loss", env.best_val_loss],
             ["best_max_l1", env.best_max_l1],
             ["loss", config.train.loss],
             ["training_size", dataset.trainX.shape[0]],
             ["validation_size", dataset.validateX.shape[0]],
             ["parameter_count", parameter_count]]
        )

    # Create visualizations
    fig_dir = run_dir.joinpath("figs")
    fig_dir.mkdir(parents=True, exist_ok=True)
    visualize_weights(model, fig_dir)
    visualize_loss(run_dir, fig_dir)
    visualize_model(run_dir, model, dataset)
    # Print run_dir
    print(run_dir)

    # Return the evaluation metric
    if config.train.evaluation_metric == 'val_loss':
        return last_info['val_loss']
    elif config.train.evaluation_metric == 'max_l1':
        return last_info['max_l1']
    else:
        raise ValueError(f"[BUG] invalid evaluation metric {config.train.evaluation_metric}")
