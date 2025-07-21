"""
Specification and validation for Configuration files.

Rest of the code can assume that the configurations are correct.
The error message here are intended for the user while in rest
of the code the assert or error message are intended for the
programmer to correct.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class DatasetConfig:
    """Specify Dataset file and h5 group name."""

    db_file: str = field(metadata={"help": ".h5 file where the input and output is dumped"})
    label: str = field(default="unknown", metadata={"help": "label of the approx block (suffix _train and _validation is added for respective splits)"})
    sample: float = field(default=1, metadata={"help": "Fraction of the dataset to use for training and validation. Default = 1 (i.e. 100%)"})

def check_member(name, val, lst):
    """Raise error if val is not in list `lst`."""
    if not val in lst:
        raise ValueError(f"Value of {name} must be on of the {lst}. {val} is not valid.")

@dataclass
class ModelConfig:
    """Configuration for Model architecture."""

    init: str = field(metadata={"help": "Initialization for weights. [u (Uniform), udd (Uniform diagonally dominant), ku (kaiming uniform), xu (Xavier uinform), default]"})
    init_param: List[float] = field(metadata={"help": "Parameters for uniform initialization function. U[-1, 1], U[-10, 10]"})
    activation: str = field(metadata={"help": "Activation function. [relu, tanh, leaky_relu]"})
    hidden_layers: List[int] = field(metadata={"help": "Hidden layers to use."})
    dropout: List[float] = field(metadata={"help": "Dropout percentage to use after activation"})
    normalize: bool = field(default=None, metadata={"help": "Normalize the input and output to mean 0, and standard deviation 1. Sets both flags normalizeX and normalizeY."})
    normalizeX: bool = field(default=None, metadata={"help": "Normalize input. Takes value from normalize."})
    normalizeY: bool = field(default=None, metadata={"help": "Normalize output. Takes value from normalize."})
    bias: bool = field(default=True, metadata={"help": "Should the linear layers have a bias term? (Default: True)"})

    def __post_init__(self):
        """Validate the config."""
        check_member('init', self.init, ['u', 'udd', 'ku', 'xu', 'default'])
        check_member('activation', self.activation, ['relu', 'tanh', 'leaky_relu'])

        if self.init == 'u' or self.init == 'udd':
            if not len(self.init_param) == 2:
                raise ValueError("TrainConfig: init_param must be [a,b] when init is u or udd")

        if self.normalize != None:
            assert(self.normalizeX == None and self.normalizeY == None)
            self.normalizeX = self.normalize
            self.normalizeY = self.normalize
        elif self.normalizeX and self.normalizeY:
            self.normalize = True
        else:
            if self.normalize == None:
                self.normalize = False
            if self.normalizeX == None:
                self.normalizeX = False
            if self.normalizeY == None:
                self.normalizeY = False

@dataclass
class OptimizerConfig:
    """Configuration for Optimizer."""

    optimizer: str = field(metadata={"help": "Optimizer to use. [rmsprop, adagrad, adamw]"})
    lr: float = field(metadata={"help": "Learning rate of Optimizer."})
    weight_decay: float = field(metadata={"help": "Optimizer weight decay"})

    def __post_init__(self):
        """Validate the config."""
        check_member('optimizer', self.optimizer, ['rmsprop', 'adagrad', 'adamw'])


@dataclass
class TrainConfig:
    """All of the training config."""

    device: str = field(metadata={"help": "Device to use. [cuda, cpu]"})

    loss: str = field(metadata={"help": "Loss function: ['mse', 'mae', 'smoothl1']"})
    epoch: int = field(metadata={"help": "Number of epochs to train."})
    batch_size: int = field(metadata={"help": "Batch size. Assigned automatically during hyperparameter tuning."})

    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        """Validate the config."""
        check_member('loss', self.loss, ['mse', 'mae', 'smoothl1'])

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    trials: int
    hidden_layers_sizes: List[int]
    n_hidden_layers: List[int]
    lr_values: List[float]
    batch_size_values: List[int]
    enable_prune: bool
    trial_id: int = 0

    def init_or_none(**kwargs):
        """Initialize Optional[TuningConfig]."""
        if not kwargs:
            return None
        else:
            return TuningConfig(**kwargs)

@dataclass
class Config:
    """Specification of config file."""

    study_name: str = field(metadata={"help": "Study name for Hyperparamter tuning"})
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tuning: Optional[TuningConfig] = field(default_factory=TuningConfig.init_or_none)
    logs_dir: Path = field(default=Path('logs'))
