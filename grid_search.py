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

def run_train(config_file, extra, dataset) -> subprocess.Popen | mp.Process:
    """Run training script."""

    extra_args = [str(arg) for arg in extra]
    if dataset:
        config = pyrallis.parse(config_class=Config, config_path=config_file, args=extra_args)
        process = mp.Process(target=train, args=(config, dataset))
        process.start()
        print(process.pid, ' '.join(extra_args))
        return process
    else:
        command_args = ["python", "main.py", "--config", config_file, *extra_args]
        process = subprocess.Popen(
            command_args,
            stdout=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        print(process.pid, ' '.join(command_args))

        return process

def check_trial(prev_trials, extra):
    checks = {}
    for i in range(0, len(extra), 2):
        key = extra[i]
        val = extra[i+1]
        checks[key[2:]] = val

    if logs.find_trial(prev_trials, checks):
        return True
    else:
        return False
def parse_field(value):
    """Try parsing value as int, float or string successively."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Hyperparameter grid search",
        description="Runs grid search on hyperparamters. Skips the trials that have already completed.",
        add_help=True
    )
    parser.add_argument('-c', '--config')
    parser.add_argument('-g', '--grid')
    parser.add_argument('-w', '--wait', action='store_true')
    parser.add_argument('-sm', '--shared-memory', action='store_true')
    args = parser.parse_args()

    config_file = args.config
    grid_file = args.grid
    shared_memory = args.shared_memory

    config = pyrallis.parse(config_class=Config, config_path=config_file, args=[])
    with open(grid_file, "r") as f:
        grid = list(csv.reader(f))
        grid = [[parse_field(v) for v in r] for r in grid]

    trials = logs.read_study(Path("./logs/").joinpath(config.study_name))

    dataset = None
    if shared_memory:
        dataset = load_shared_dataset(config.dataset)

    processes: list[subprocess.Popen | mp.Process] = []
    def rec(grid, params):
        """Recursively explore the grid and at the end run_train."""
        if len(grid) == 0:
            if check_trial(trials, params):
                print("Skipped ", ' '.join([str(arg) for arg in params]))
                return
            else:
                p = run_train(config_file, params, dataset)
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
