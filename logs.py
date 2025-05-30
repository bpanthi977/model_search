from pathlib import Path
import csv
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import pyrallis

from config import Config, TrainConfig

@dataclass
class Trial:

    name: str
    config: Config
    train_loss: List[Tuple[int, float]]
    val_loss: List[Tuple[int, float]]

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

    if len(train_loss) == 0 or len(val_loss) == 0:
        return False
    return Trial(name=trial_dir.name, config=config, train_loss=train_loss, val_loss=val_loss)

def read_study(study_dir: Path):
    trials = []
    for d in study_dir.iterdir():
        if d.is_dir():
            t = read_run(d)
            if t:
                trials.append(t)

    return trials

def get_value(config, key):
    def rec(config, keys):
        if len(keys) == 0:
            return config
        else:
            return rec(config.__dict__[keys[0]], keys[1:])
    return rec(config, key.split('.'))

def find_trial(trials, checks):
    for t in trials:
        match = True
        for (key, value) in checks.items():
            if get_value(t.config, key) != value:
                match = False
                break

        if match == True:
            return t

def create_df(trials):
    df_loss = pd.DataFrame(columns=["name", "epoch", "train_loss", "val_loss"])
    df_config = pd.DataFrame(columns=["name", "loss", "batch_size",
                                      "init", "activation", "hidden_layers", "dropout",
                                      "optimizer", "lr", "weight_decay"])

    for t in trials:
        for (tl, vl) in zip(t.train_loss, t.val_loss):
            epoch = tl[0]
            df_loss.loc[len(df_loss)] = [t.name, epoch, tl[1], vl[1]]

        train = t.config.train
        model = t.config.train.model
        optim = t.config.train.optim
        df_config.loc[len(df_config)] = [t.name, train.loss, train.batch_size,
                                         model.init, model.activation, model.hidden_layers, model.dropout,
                                         optim.optimizer, optim.lr, optim.weight_decay]

    df = df_config.merge(df_loss, how='inner', validate='one_to_many')
    return df
