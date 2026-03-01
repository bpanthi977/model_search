from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import torch.nn as nn
from typing import Optional

from config import Config, Checkpoint

def get_loss_fn(loss: str):
    """Create loss function."""
    if loss == 'mse':
        return nn.MSELoss(reduction='mean')
    elif loss == 'mae':
        return nn.L1Loss(reduction='mean')
    elif loss == 'smoothl1':
        return nn.SmoothL1Loss(reduction='mean', beta=1.0)

    assert False, f"Impossible value of loss function {loss}. Fix validate in TrainConfig."

def extract_hparams(config: Config, checkpoint: Optional[Checkpoint]):
    """Extracts hyperparameters from Config object."""
    hparams_dict = {
        "study_name": config.study_name,
        "dataset/file": config.dataset.db_file,
        "dataset/subset": config.dataset.subset,
        "dataset/sample": config.dataset.sample,
        "model/init": config.train.model.init,
        "model/activation": config.train.model.activation,
        "model/normalize": config.train.model.normalize,
        "optimizer": config.train.optim.optimizer,
        "lr": config.train.optim.lr,
        "loss": config.train.loss,
        "weight_decay": config.train.optim.weight_decay,
        "batch_size": config.train.batch_size,
        "epochs": config.train.epoch
    }
    
    if config.train.model.hidden_layers:
         hparams_dict["model/hidden_layers"] = str(config.train.model.hidden_layers)

    if checkpoint:
        hparams_dict["continue_from"] = checkpoint['continue_from']
        hparams_dict["continue_epoch"] = checkpoint['epoch']
        hparams_dict["continue_lr_scheduler_state"] = True if checkpoint['lr_scheduler_state_dict'] else False
        hparams_dict["continue_optimizer_state"] = True if checkpoint['optimizer_state_dict'] else False

    return hparams_dict

def log_hparams(writer: SummaryWriter, hparams_dict: dict, metrics: dict, global_step: int):
    """
    Logs hyperparameters and metrics to the main event file of the SummaryWriter.
    Consolidates hparams into the same file as scalar metrics.
    """
    # Generate the hparams summary
    exp, ssi, sei = hparams(hparams_dict, metrics)
    
    # Write to the file writer directly
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metrics.items():
        writer.add_scalar(k, v, global_step)
