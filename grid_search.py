"""Run hyperparameter gridsearch for HGF."""
import subprocess
from pathlib import Path
import pyrallis
import argparse
import csv

import logs
from config import Config

def run_train(config_file, extra):
    """Run training script."""
    command_args = ["python", "main.py", "--config", config_file, *[str(arg) for arg in extra]]
    process = subprocess.Popen(
        command_args,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(process.pid, ' '.join(command_args))


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Hyperparameter grid search",
        description="Runs grid search on hyperparamters. Skips the trials that have already completed.",
        add_help=True
    )
    parser.add_argument('-c', '--config')
    parser.add_argument('-g', '--grid')
    args = parser.parse_args()

    print(args)
    config_file = args.config
    grid_file = args.grid

    config = pyrallis.parse(config_class=Config, config_path=config_file, args=[])
    with open(grid_file, "r") as f:
        grid = list(csv.reader(f))

    trials = logs.read_study(Path("./logs/").joinpath(config.study_name))

    def rec(grid, params):
        """Recursively explore the grid and at the end run_train."""
        if len(grid) == 0:
            if check_trial(trials, params):
                print("Skipped ", ' '.join([str(arg) for arg in params]))
                return
            else:
                return run_train(config_file, params)


        param = grid[0]
        other_params = grid[1:]

        flag = param[0]
        values = param[1:]
        for value in values:
            rec(other_params, [flag, value, *params])

    rec(grid, [])
