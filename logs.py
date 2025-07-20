from pathlib import Path
import csv
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pyrallis

from config import Config, TrainConfig

@dataclass
class Timing:
    start: str
    end: str
    total_time_min: float

@dataclass
class Trial:

    name: str
    config: Config
    train_loss: List[Tuple[int, float]]
    val_loss: List[Tuple[int, float]]
    time: Optional[Timing]

    @property
    def final_val_loss(self):
        return self.val_loss[len(self.val_loss) - 1][1]

    @property
    def epochs(self):
        return min(len(self.train_loss), len(self.val_loss))

    def __repr__(self):
        return f"Trial(name={self.name}\n\tconfig={self.config},\n\ttrain_loss=[..., {self.train_loss[-1]}],\n\tval_loss=[..., {self.val_loss[-1]}])"

def read_run(trial_dir: Path):
    config_file = trial_dir.joinpath("config.yaml")
    if not config_file.exists():
        return False

    config = pyrallis.parse(config_class=Config, config_path=config_file, args=[])
    with open(trial_dir.joinpath("train_loss.csv"), "r") as f:
        train_loss = [[int(epoch), float(loss), float(time)] for [epoch, loss, time] in list(csv.reader(f))]

    with open(trial_dir.joinpath("val_loss.csv"), "r") as f:
        val_loss = [[int(epoch), float(loss), float(time)] for [epoch, loss, time] in list(csv.reader(f))]

    info_file = trial_dir.joinpath("info.csv")
    time = None
    if info_file.exists():
        with open(info_file, "r") as f:
            entries = list(csv.reader(f))
            total_time = entries[2][1]
            minutes = 0
            factor = 60
            for part in total_time.split(':'):
                minutes += int(part) * factor
                factor = factor / 60

            time = Timing(start=entries[0][1], end=entries[1][1], total_time_min=minutes)

    if len(train_loss) == 0 or len(val_loss) == 0:
        return False
    return Trial(name=trial_dir.name, config=config, train_loss=train_loss, val_loss=val_loss, time=time)

def read_study(study_dir: Path):
    trials = []
    if not study_dir.exists():
        return []

    for d in study_dir.iterdir():
        if d.is_dir():
            t = read_run(d)
            if t:
                trials.append(t)

    return trials

def create_df(trials):
    df_loss = pd.DataFrame(columns=["name", "epoch", "train_loss", "val_loss"])
    df_config = pd.DataFrame(columns=["name", "loss", "batch_size", "sample",
                                      "init", "init_param", "activation", "hidden_layers", "dropout", "bias", "normalize",
                                      "optimizer", "lr", "weight_decay",
                                      "train_time"])

    for t in trials:
        for (tl, vl) in zip(t.train_loss, t.val_loss):
            epoch = tl[0]
            df_loss.loc[len(df_loss)] = [t.name, epoch, tl[1], vl[1]]

        train = t.config.train
        model = t.config.train.model
        optim = t.config.train.optim
        df_config.loc[len(df_config)] = [t.name, train.loss, train.batch_size, t.config.dataset.sample,
                                         model.init, model.init_param, model.activation, model.hidden_layers, model.dropout, model.bias, model.normalize,
                                         optim.optimizer, optim.lr, optim.weight_decay,
                                         t.time.total_time_min if t.time else 0]

    df = df_config.merge(df_loss, how='inner', validate='one_to_many')
    return df
