"""Central CLI for all the functionality of the project."""
import argparse
from h5py import Dataset
import pyrallis
from datetime import datetime
import random
import string
from typing import Optional

from train import train_log
from tune import tune
from config import Config

def train(config: Config, dataset: Optional[Dataset]):
    ## Generate a random suffix so that train started at same time don't clash
    random_suffix = ''.join([random.choice(string.ascii_letters) for i in range(4)])
    trial_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + random_suffix

    loss = train_log(config, trial_id=trial_name, callbacks=[], dataset=dataset)
    print(f"Loss = {loss}")

if __name__ == "__main__":
    # Parse argument, config and show help
    parser = argparse.ArgumentParser(
        prog="Model Tuning",
        description="Trains and tunes hyperparameters of MLP model",
        add_help=False
    )
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-c', '--config')
    parser.add_argument('--tune', action='store_true')

    (args, args_rest) = parser.parse_known_args()
    if args.help:
        parser.print_help()
        config_parser = pyrallis.argparsing.ArgumentParser(config_class=Config)
        config_parser.print_help()
        exit(0)

    try:
        config = pyrallis.parse(config_class=Config, config_path=args.config, args=args_rest)
    except pyrallis.ParsingError as e:
        print(f"Failure to parse config file. Details:\n{e}")
        exit(1)

    # Run train or tuning
    if not args.tune: # just train
        train(config, None)
    else:
        tune(config)

    exit(0)
