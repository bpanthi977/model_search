from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

@dataclass
class DatasetConfig:
    db_file: str = field(metadata={"help": ".h5 file where the input and output is dumped"})
    label: str = field(default="unknown", metadata={"help": "label of the approx block (suffix _train and _validation is added for respective splits)"})

def check_member(name, val, lst):
    if not val in lst:
        raise ValueError(f"Value of {name} must be on of the {lst}. {val} is not valid.")

@dataclass
class ModelConfig:
    init: str = field(metadata={"help": "Initialization for weights. [u (Uniform), udd (Uniform diagonally dominant), ku (kaiming uniform), xu (Xavier uinform)]"})
    init_param: List[float] = field(metadata={"help": "Parameters for uniform initialization function. U[-1, 1], U[-10, 10]"})
    activation: str = field(metadata={"help": "Activation function. [relu, tanh, leaky_relu]"})
    hidden_layers: List[int] = field(metadata={"help": "Hidden layers to use."})
    dropout: List[int] = field(metadata={"help": "Dropout percentage to use after activation"})

    def __post_init__(self):
        check_member('init', self.init, ['u', 'udd', 'ku', 'xu'])
        check_member('activation', self.activation, ['relu', 'tanh', 'leaky_relu'])

        if self.init == 'u' or self.init == 'udd':
            if not len(self.init_param) == 2:
                raise ValueError("TrainConfig: init_param must be [a,b] when init is u or udd")


@dataclass
class OptimizerConfig:
    optimizer: str = field(metadata={"help": "Optimizer to use. [rmsprop, adagrad, adamw]"})
    lr: float = field(metadata={"help": "Learning rate of Optimizer."})
    weight_decay: float = field(metadata={"help": "Optimizer weight decay"})

    def __post_init__(self):
        check_member('optimizer', self.optimizer, ['rmsprop', 'adagrad', 'adamw'])


@dataclass
class TrainConfig:
    device: str = field(metadata={"help": "Device to use. [cuda, cpu]"})

    loss: str = field(metadata={"help": "Loss function: ['mse', 'mae', 'smoothl1']"})
    epoch: int = field(metadata={"help": "Number of epochs to train."})
    batch_size: int = field(metadata={"help": "Batch size. Assigned automatically during hyperparameter tuning."})

    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        check_member('loss', self.loss, ['mse', 'mae', 'smoothl1'])

@dataclass
class TuningConfig:
    trials: int
    hidden_layers_sizes: List[int]
    n_hidden_layers: List[int]
    lr_values: List[float]
    batch_size_values: List[int]
    enable_prune: bool
    trial_id: int = 0

    def init_or_none(**kwargs):
        if not kwargs:
            return None
        else:
            return TuningConfig(**kwargs)

@dataclass
class Config:
    study_name: str = field(metadata={"help": "Study name for Hyperparamter tuning"})
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tuning: Optional[TuningConfig] = field(default_factory=TuningConfig.init_or_none)
    logs_dir: Path = field(default=Path('logs'))
