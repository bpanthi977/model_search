from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

@dataclass
class DatasetConfig:
    db_file: str = field(metadata={"help": ".h5 file where the input and output is dumped"})
    label: str = field(default="unknown", metadata={"help": "label of the approx block"})

@dataclass
class TrainConfig:
    activation: str = field(metadata={"help": "Activation function. [relu]"})
    device: str = field(metadata={"help": "Device to use. [cuda, cpu]"})
    optimizer: str = field(metadata={"help": "Optimizer to use. [adam]"})
    epoch: int = field(metadata={"help": "Number of epochs to train."})
    hidden_layers: List[int] = field(default_factory=lambda:[], metadata={"help": "Hidden layers to use. Assigned automatically during hyperparameter tuning."})
    lr: float = field(default=0.001, metadata={"help": "Learning rate. Assigned automatically during hyperparameter tuning."})
    batch_size: int = field(default=1024, metadata={"help": "Batch size. Assigned automatically during hyperparameter tuning."})


@dataclass
class TuningConfig:
    trials: int
    hidden_layers_sizes: List[int]
    n_hidden_layers: List[int]
    lr_values: List[float]
    batch_size_values: List[int]
    trial_id: int = 0

@dataclass
class Config:
    study_name: str = field(metadata={"help": "Study name for Hyperparamter tuning"})
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    logs_dir: Path = field(default=Path('logs'))
