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
from typing import Optional, cast
import torch
import numpy as np

from train import train_log, save_checkpoint
from tune import tune
from config import Config, Checkpoint
from validate_model import create_model_from_checkpoint, validate
from lr_tune import run_lr_tune

def redirect_output(path: Path):
    log_file = open(path, 'a')

    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate the file descriptor for stdout (1) and stderr (2)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

def train(config: Config, dataset: Optional[Dataset], redirect_io: bool, checkpoint_arg: tuple[Optional[Path], bool, int]):
    config = copy.deepcopy(config)
    config.tuning = None

    ## Generate a random suffix so that train started at same time don't clash
    seed = int.from_bytes(os.urandom(8), byteorder='big')
    rng = random.Random(seed)
    random_suffix = ''.join([rng.choice(string.ascii_letters) for i in range(4)])
    trial_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + random_suffix
    run_dir = config.logs_dir.joinpath(config.study_name).joinpath(f"{trial_name}")

    checkpoint_path, use_only_weights, checkpoint_every_n = checkpoint_arg
    checkpoint: Optional[Checkpoint] = None
    if checkpoint_path:
        shutil.copytree(checkpoint_path, run_dir)
        with open(run_dir.joinpath("continue_from"), "w") as f:
            f.write(checkpoint_path.name +'\n' + f'use_only_weights={use_only_weights}')

        # Restore checkpoint
        checkpoint_pth = run_dir.joinpath('checkpoint.pth')
        if checkpoint_pth.exists():
            checkpoint = cast(Checkpoint, torch.load(checkpoint_pth))
            checkpoint['continue_from'] = checkpoint_path.name
            if use_only_weights:
                checkpoint['lr_scheduler_state_dict'] = None
                checkpoint['optimizer_state_dict'] = None

    run_dir.mkdir(parents=True, exist_ok=True)
    print(run_dir)
    if redirect_io:
        redirect_output(run_dir.joinpath("stdout.txt"))

    def checkpoint_every_callback(env, info):
        epoch = info['epoch'] # starts from 0
        if (epoch + 1) % checkpoint_every_n == 0:
            save_checkpoint(env, run_dir.joinpath(f"checkpoint-{epoch + 1}.pth"))

    callbacks = []
    if checkpoint_every_n != 0:
        callbacks.append(checkpoint_every_callback)

    evaluation_metric_value = train_log(config, trial_id=trial_name, callbacks=callbacks, dataset=dataset, checkpoint=checkpoint)
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
    parser.add_argument('--checkpoint-use-only-weights', action='store_true')
    parser.add_argument('--checkpoint-every-n')
    parser.add_argument('--lr-tune', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--create-model', action='store_true')
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
    if not (args.config or args.checkpoint):
        print("Specify either --config or --checkpoint.")
        exit(1)

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
            changable_flags = ['train.epoch', 'dataset.db_file', 'study_name']
            if arg[2:] not in changable_flags:
                print(f"You can only override f{','.join(changable_flags)} arguments. Can't change {arg}.")
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
    elif args.create_model:
        if not checkpoint_path:
            print(f"Specify --checkpoint for --create-model.")
            exit(1)
        create_model_from_checkpoint(config, checkpoint_path)
    elif args.lr_tune:
        run_lr_tune(config)
    elif not args.tune: # just train
        checkpoint_every_n = 0
        if args.checkpoint_every_n:
            checkpoint_every_n = int(args.checkpoint_every_n)
        train(config, None, args.log_stdout, (checkpoint_path, args.checkpoint_use_only_weights, checkpoint_every_n))
    else:
        tune(config, args.postgres)

    exit(0)
