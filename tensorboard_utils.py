from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from config import Config

def extract_hparams(config: Config):
    """Extracts hyperparameters from Config object."""
    hparams_dict = {
        "study_name": config.study_name,
        "dataset": config.dataset.db_file,
        "model_init": config.train.model.init,
        "model_activation": config.train.model.activation,
        "normalize": config.train.model.normalize,
        "optimizer": config.train.optim.optimizer,
        "loss": config.train.loss,
        "weight_decay": config.train.optim.weight_decay,
        "batch_size": config.train.batch_size,
    }
    
    if config.train.model.hidden_layers:
         hparams_dict["hidden_layers"] = str(config.train.model.hidden_layers)
         
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
