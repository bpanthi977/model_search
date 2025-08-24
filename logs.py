from pathlib import Path
import csv
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

import argparse
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
    global total_parse_time
    config_file = trial_dir.joinpath("config.yaml")
    if not config_file.exists():
        return False

    config = pyrallis.parse(config_class=Config, config_path=config_file, args=[])
    with open(trial_dir.joinpath("train_loss.csv"), "r") as f:
        def parse_row(row):
            lr = row[3] if len(row) >= 4 else config.train.optim.lr
            # epoch, loss, time, lr
            return [int(row[0]), float(row[1]), float(row[2]), lr]
        train_loss = [parse_row(row) for row in list(csv.reader(f))]

    with open(trial_dir.joinpath("val_loss.csv"), "r") as f:
        def parse_row(row):
            max_l1 = float(row[3]) if len(row) >= 4 else None
            return [int(row[0]), float(row[1]), float(row[2]), max_l1]
        # epoch, loss, time, max_l1
        val_loss = [parse_row(row) for row in list(csv.reader(f))]

    info_file = trial_dir.joinpath("info.csv")
    time = None
    if info_file.exists():
        with open(info_file, "r") as f:
            entries = list(csv.reader(f))
            if len(entries) < 3:
                return False

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
    loss_rows = []
    config_rows = []
    for t in trials:
        for (tl, vl) in zip(t.train_loss, t.val_loss):
            epoch = tl[0]
            loss_rows.append([t.name, epoch, tl[1], vl[1], vl[3]])

        train = t.config.train
        model = t.config.train.model
        optim = t.config.train.optim
        config_rows.append([t.name, train.loss, train.batch_size, t.config.dataset.sample,
                                         model.init, model.init_param, model.activation, model.hidden_layers, model.dropout, model.bias, model.normalizeX, model.normalizeY, model.batchnorm,
                                         optim.optimizer, optim.lr, optim.weight_decay,
                                         t.time.total_time_min if t.time else 0])

    df_loss = pd.DataFrame(loss_rows, columns=["name", "epoch", "train_loss", "val_loss", "val_max_l1"])
    df_config = pd.DataFrame(config_rows, columns=["name", "loss", "batch_size", "sample",
                                                   "init", "init_param", "activation", "hidden_layers", "dropout", "bias", "normalizeX", "normalizeY", "batchnorm",
                                                   "optimizer", "lr", "weight_decay",
                                                   "train_time"])
    df = df_config.merge(df_loss, how='inner', validate='one_to_many')
    return df

def describe_list(hl):
    hl_same = len(set(hl)) <= 1
    if hl_same:
        return str(hl[0]) + '*' + str(len(hl))
    else:
        return str(hl)

def describe_row(row):
    desc = ''
    desc += describe_list(row['hidden_layers'])
    if row['batchnorm']:
        desc = f'|{desc}|_b'
    if not row['bias']:
        desc += '!b'
    if len(row['dropout']) != 0:
        desc += '-' + describe_list(row['dropout'])
    if row['activation'] != 'relu':
        desc += '-' + row['activation']
    if row['batch_size'] != 1024:
        desc += '-bs(' + str(row['batch_size']) + ')'
    if row['optimizer'] != 'adagrad' or row['lr'] != '0.01' or row['weight_decay'] != 0.0:
        opt = {'adagrad': 'ag', 'rmsprop': 'rp', 'adamw': 'am'}
        desc += f'-{opt[row['optimizer']]}({row['lr']:.6}'
        if row['weight_decay'] != 0.0:
            desc += f',{row['weight_decay']}'
        desc += ')'
    if row['normalizeX'] or row['normalizeY']:
        desc = f'|{desc}|_'
        if row['normalizeX']:
            desc += 'x'
        if row['normalizeY']:
            desc += 'y'
    if row['loss'] != 'mae':
        if row['loss'] == 'mse':
            desc += "_l2"
        elif row['loss'] == 'smoothl1':
            desc += '_sl1'

    return desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Logs"
    )
    parser.add_argument('--logs')
    parser.add_argument('--sort-by', help="Options: [loss, name]. --sort-by loss sorts the results in increasing order of validation loss.")
    parser.add_argument('--save', help="Save the summary to summary.txt file in the logs folder.", action='store_true')

    args = parser.parse_args()
    if not args.logs:
        print("Specify logs directory: --logs")
        exit(1)

    logs_dir = Path(args.logs)
    if not logs_dir.exists():
        print(f"Logs directory {logs_dir} doesn't exists.")
        exit(1)

    study = read_study(logs_dir)
    df_all = create_df(study)


    final_rows = df_all.loc[df_all.groupby('name')['epoch'].idxmax()]
    final_rows = final_rows.set_index('name')
    best_rows = df_all.groupby('name')['val_loss'].idxmin()

    final_rows['val_loss_best'] = df_all.loc[best_rows].set_index('name')['val_loss']
    final_rows['best_epoch'] = df_all.loc[best_rows].set_index('name')['epoch']
    final_rows['desc'] = final_rows.apply(describe_row, axis=1)

    if args.sort_by == 'loss':
        final_rows = final_rows.sort_values(by='val_loss_best')
    else:
        final_rows = final_rows.sort_values(by='name')

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.expand_frame_repr", False)

    if args.save:
        summary_file = logs_dir.joinpath("summary.txt")
        with open(summary_file, 'r') as f:
            f.write(str(final_rows))

        print(f"Summary saved at: {summary_file}")
    else:
        print(final_rows)
