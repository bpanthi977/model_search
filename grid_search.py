"""Run hyperparameter gridsearch for HGF."""
from multiprocessing import Process
import subprocess
from pathlib import Path
import pyrallis
import argparse
import csv
import torch.multiprocessing as mp

import logs
from config import Config, DatasetConfig
from dataset import load_dataset, Dataset
from main import train

def load_shared_dataset(config: DatasetConfig):
    dataset = load_dataset(config)
    return Dataset(dataset.trainX.share_memory_(), dataset.trainY.share_memory_(),
                   dataset.validateX.share_memory_(), dataset.validateY.share_memory_())

def start_subprocess(config_file, extra) -> subprocess.Popen:
    """Run training script."""

    extra_args = [str(arg) for arg in extra]
    command_args = ["python", "main.py", "--log-stdout", "--config", config_file, *extra_args]
    process = subprocess.Popen(
        command_args,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(process.pid, ' '.join(command_args))

    return process

def parse_field(value):
    """Try parsing value as int, float or string successively."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def find_trial(trials: list[logs.Trial], trial: Config):
    for t in trials:
        if t.config.train == trial.train and t.config.dataset == trial.dataset:
            return True

    return False

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(
        prog="Hyperparameter grid search",
        description="Runs grid search on hyperparamters. Skips the trials that have already completed.",
        add_help=True
    )
    parser.add_argument('-c', '--config')
    parser.add_argument('-g', '--grid')
    parser.add_argument('-w', '--wait', action='store_true')
    parser.add_argument('-sm', '--shared-memory', action='store_true')
    parser.add_argument('--dry-run', action="store_true")
    args = parser.parse_args()

    config_file = args.config
    grid_file = args.grid
    shared_memory = args.shared_memory
    dry_run = args.dry_run

    config = pyrallis.parse(config_class=Config, config_path=config_file, args=[])
    with open(grid_file, "r") as f:
        grid = list(csv.reader(f, skipinitialspace=True))
        grid = [[parse_field(v) for v in r] for r in grid]

    trials = logs.read_study(Path("./logs/").joinpath(config.study_name))

    dataset = None
    if shared_memory:
        dataset = load_shared_dataset(config.dataset)

    processes: list[subprocess.Popen | mp.Process] = []

    def rec(grid, params):
        """Recursively explore the grid and at the end run_train."""
        if len(grid) == 0:
            trial_config = pyrallis.parse(config_class=Config, config_path=config_file, args=[str(arg) for arg in params])

            if find_trial(trials, trial_config):
                print("Skipped ", ' '.join([str(arg) for arg in params]))
                return
            elif shared_memory:
                print(' '.join([str(arg) for arg in params]))
                if dry_run:
                    return

                p = mp.Process(target=train, args=(trial_config, dataset, True))
                p.start()
                processes.append(p)
                return
            else:
                if dry_run:
                    print(' '.join([str(arg) for arg in params]))
                    return

                p = start_subprocess(config_file, params)
                processes.append(p)
                return

        param = grid[0]
        other_params = grid[1:]

        flag = param[0]
        values = param[1:]
        for value in values:
            rec(other_params, [flag, value, *params])

    rec(grid, [])

    if args.wait:
        for p in processes:
            if isinstance(p, mp.Process):
                p.join()
            else:
                p.wait()
