"""
Specification and validation for Configuration files.

Rest of the code can assume that the configurations are correct.
The error message here are intended for the user while in rest
of the code the assert or error message are intended for the
programmer to correct.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import re
import traceback

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

def parse_hidden_layers(hidden_layers: List[Union[str, int]]):
    result = []
    # MULT is deprecated after commit 9120d0a
    layer_types = {'linear', 'NALU', 'NALUi1', 'NALUi2', 'MULT0', 'MULT'}
    layer_act = {'I'}
    regex_match_types = '|'.join(layer_types)
    regex_match_act = "|".join(layer_act)
    for i, hl in enumerate(hidden_layers):
        if isinstance(hl, int):
            result.append(('linear', hl))
        elif isinstance(hl, str):
            if match := re.match(rf'({regex_match_types})\(([0-9]+)\)(?:-({regex_match_act}))?', hl):
                hl_type, size, act = match.groups()
                size = int(size)
                assert size != 0
                result.append((hl_type, size, act))
            elif hl in layer_types:
                if i + 1 != len(hidden_layers):
                    raise ValueError(f"Invalid hidden_layer {hl}: type without size is only allowed as last layer")
                result.append((hl, 0, None))
            elif match := re.match('([0-9]+)', hl):
                size = int(match.groups()[0])
                assert size != 0
                result.append(('linear', size, None))
            else:
                raise ValueError(f"Invalid hidden_layer specification {hl}")
    return result

@dataclass
class ModelConfig:
    """Configuration for Model architecture."""

    init: str = field(metadata={"help": "Initialization for weights. [u (Uniform), udd (Uniform diagonally dominant), ku (kaiming uniform), xu (Xavier uinform), default]"})
    init_param: List[float] = field(metadata={"help": "Parameters for uniform initialization function. U[-1, 1], U[-10, 10]"})
    activation: str = field(metadata={"help": "Activation function. [relu, tanh, leaky_relu]"})
    hidden_layers: List[Union[str, int]] = field(metadata={"help": "Hidden layers to use."})
    dropout: List[float] = field(metadata={"help": "Dropout percentage to use after activation"})
    normalize: bool = field(default=None, metadata={"help": "Deprecated. True implies both normalizeX and normalizeY are true."})
    normalizeX: bool = field(default=None, metadata={"help": "Normalize input."})
    normalizeY: bool = field(default=None, metadata={"help": "Normalize output."})
    batchnorm: bool = field(default=False, metadata={"help": "Enable batchnorm in each hidden layer."})
    bias: bool = field(default=True, metadata={"help": "Should the linear layers have a bias term? (Default: True)"})

    def __post_init__(self):
        """Validate the config."""
        check_member('init', self.init, ['u', 'udd', 'ku', 'xu', 'default'])
        check_member('activation', self.activation, ['relu', 'tanh', 'leaky_relu'])

        if self.init == 'u' or self.init == 'udd':
            if not len(self.init_param) == 2:
                raise ValueError("TrainConfig: init_param must be [a,b] when init is u or udd")

        if self.normalize == True:
            assert self.normalizeX == self.normalizeY == None or self.normalizeX == self.normalizeY == True, "normalize and normalizeX,Y field values don't match."
            self.normalizeX = self.normalizeY = True

        if self.normalizeX == None:
            self.normalizeX = False
        if self.normalizeY == None:
            self.normalizeY = False
        if self.normalize == None:
            self.normalize = self.normalizeX and self.normalizeY

        parse_hidden_layers(self.hidden_layers)

def parse_lr_scheduler(lr: str):
    """Parse lr and return type, initial_lr, *other_params"""
    if (match := re.match('([0-9\\.]+)', lr)):
        start: float = float(match.groups()[0])
        return 'constant', start
    elif (match := re.match('linear\\(([0-9\\.]+)-([0-9\\.]+)\\)', lr)):
        start, end = match.groups()
        start = float(start)
        end = float(end)
        return 'linear', start, end
    else:
        raise Exception(f"[BUG] LR Schedule not supported '{lr}'")

@dataclass
class OptimizerConfig:
    """Configuration for Optimizer."""

    optimizer: str = field(metadata={"help": "Optimizer to use. [rmsprop, adagrad, adamw]"})
    lr: str = field(metadata={"help": "Learning rate of Optimizer. Can be a string."})
    nalu_lr: Optional[float] = field(metadata={"help": "Learning rate for NALU layers."})
    weight_decay: float = field(metadata={"help": "Optimizer weight decay"})

    def __post_init__(self):
        """Validate the config."""
        check_member('optimizer', self.optimizer, ['rmsprop', 'adagrad', 'adamw'])
        try:
            parse_lr_scheduler(self.lr)
        except Exception as e:
            traceback.print_exc()
            raise e

@dataclass
class TrainConfig:
    """All of the training config."""

    device: str = field(metadata={"help": "Device to use. [cuda, cpu]"})

    loss: str = field(metadata={"help": "Loss function: ['mse', 'mae', 'smoothl1']"})
    epoch: int = field(metadata={"help": "Number of epochs to train."})
    batch_size: int = field(metadata={"help": "Batch size. Assigned automatically during hyperparameter tuning."})
    validation_interval: int = field(default=1, metadata={"help": "Get validation loss every `validation_interval` epochs."})
    evaluation_metric: str = field(default='val_loss', metadata={'help': 'The evaluation metric to choose the best checkpoint and tuning. [val_loss, max_l1]'})

    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        """Validate the config."""
        check_member('loss', self.loss, ['mse', 'mae', 'smoothl1'])
        check_member('evaluation_metric', self.evaluation_metric, ['val_loss', 'max_l1'])

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    trials: int
    tune_normalize: bool
    enable_prune: bool
    batch_size_values: List[int]
    hidden_layers_size_range: Optional[List[int]] = None
    n_hidden_layers: Optional[List[int]] = None
    hidden_layer_types: Optional[List[str]] = None
    lr_range: Optional[List[float]] = None
    optimizer: Optional[List[str]] = None
    weight_decay_range: Optional[List[float]] = None
    trial_id: int = 0

    def init_or_none(**kwargs):
        """Initialize Optional[TuningConfig]."""
        if not kwargs:
            return None
        else:
            return TuningConfig(**kwargs)

    def __post_init__(self):
        try:
            if self.lr_range:
                lr_range = self.lr_range
                assert len(lr_range) == 2, "lr_range must be two floats (low, high)"
                assert self.lr_range[0] <= self.lr_range[1], f"lr_range must be (low, high) but {lr_range[0]} > {lr_range[1]}"

            hl_range = self.hidden_layers_size_range
            if hl_range:
                assert len(hl_range) == 2, "hidden_layers_size_range must be two floats (low, high)"
                assert hl_range[0] <= hl_range[1], f"hidden_layers_size_range must be (low, high) but {hl_range[0]} > {hl_range[1]}"

            wd_range = self.weight_decay_range
            if wd_range:
                assert len(wd_range) == 2, "weight_decay_range must be two floats (low, high)"
                assert wd_range[0] <= wd_range[1], f"weight_decay_range must be (low, high) but {wd_range[0]} > {wd_range[1]}"

            if self.optimizer:
                for optimizer in self.optimizer:
                    assert (optimizer in ['adamw', 'adagrad', 'rmsprop']), "optimizer must be one of ['adamw', 'adagrad', 'rmsprop']"
        except AssertionError as e:
            print(f"Error: {e}")
            raise e

@dataclass
class Config:
    """Specification of config file."""

    study_name: str = field(metadata={"help": "Study name for Hyperparamter tuning"})
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tuning: Optional[TuningConfig] = field(default_factory=TuningConfig.init_or_none)
    logs_dir: Path = field(default=Path('logs'))
