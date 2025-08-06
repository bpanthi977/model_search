"""Central CLI for all the functionality of the project."""
import argparse
import os
import sys
import shutil
from pathlib import Path
import copy
from h5py import Dataset
import pyrallis
from datetime import datetime
import random
import string
from typing import Optional
import torch
import numpy as np

from train import train_log
from tune import tune
from config import Config
from validate_model import validate

def redirect_output(path: Path):
    log_file = open(path, 'a')

    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate the file descriptor for stdout (1) and stderr (2)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

def train(config: Config, dataset: Optional[Dataset], redirect_io: bool, checkpoint_path: Optional[Path]):
    config = copy.deepcopy(config)
    config.tuning = None

    ## Generate a random suffix so that train started at same time don't clash
    seed = int.from_bytes(os.urandom(8), byteorder='big')
    rng = random.Random(seed)
    random_suffix = ''.join([rng.choice(string.ascii_letters) for i in range(4)])
    trial_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + random_suffix
    run_dir = config.logs_dir.joinpath(config.study_name).joinpath(f"{trial_name}")

    if checkpoint_path:
        shutil.copytree(checkpoint_path, run_dir)
        with open(run_dir.joinpath("continue_from"), "w") as f:
            f.writelines([checkpoint_path.name])

    run_dir.mkdir(parents=True, exist_ok=True)
    print(run_dir)
    if redirect_io:
        redirect_output(run_dir.joinpath("stdout.txt"))

    evaluation_metric_value = train_log(config, trial_id=trial_name, callbacks=[], dataset=dataset)
    print(f"{config.train.evaluation_metric} = {evaluation_metric_value}")

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Parse argument, config and show help
    parser = argparse.ArgumentParser(
        prog="Model Tuning",
        description="Trains and tunes hyperparameters of MLP model",
        add_help=False
    )
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-c', '--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('-l', '--log-stdout', action='store_true')
    parser.add_argument('--seed')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--postgres', action='store_true')

    (args, args_rest) = parser.parse_known_args()
    if args.help:
        parser.print_help()
        config_parser = pyrallis.argparsing.ArgumentParser(config_class=Config)
        config_parser.print_help()
        exit(0)

    set_all_seeds(int(args.seed or '42'))

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint path doesn't exits {checkpoint_path}")
            exit(1)

        if args.config:
            print(f"Can't specify both --config and --checkpoint. Use one.")
            exit(1)

        for i in range(0, len(args_rest), 2):
            arg = args_rest[i]
            if arg != "--train.epoch":
                print(f"You can only override --train.epoch argument. Can't change {arg}.")
                exit(1)

    try:
        config_path = args.config or checkpoint_path.joinpath('config.yaml')
        config = pyrallis.parse(config_class=Config, config_path=config_path, args=args_rest)
    except pyrallis.ParsingError as e:
        print(f"Failure to parse config file. Details:\n{e}")
        exit(1)


    # Run train or tuning
    if args.validate:
        if not checkpoint_path:
            print(f"Specify --checkpoint for --validate.")
            exit(1)

        validate(config, checkpoint_path)
    elif not args.tune: # just train
        train(config, None, args.log_stdout, checkpoint_path)
    else:
        tune(config, args.postgres)

    exit(0)
