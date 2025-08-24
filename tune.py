"""Hyperparameter tuning."""
import pickle
import dataclasses
import optuna
from pathlib import Path
from datetime import datetime
import copy
import math
from typing import List

import pyrallis

from dataset import load_dataset
from train import train_log
from config import Config, TrainConfig

def suggest_layers(study: optuna.Trial, config: Config, label:str, choices: List[int]):
    layers = []
    layer_sizes = []
    n = study.suggest_categorical(label, choices)
    for i in range(n):
        hl_range = config.tuning.hidden_layers_size_range
        hl_size = study.suggest_int(f"{label}_sizes{i}", hl_range[0], hl_range[1])
        hl_type = study.suggest_categorical(f"{label}_type{i}", config.tuning.hidden_layer_types)
        layers.append(f"{hl_type}({hl_size})")
        layer_sizes.append(hl_size)

    return layers, layer_sizes


def create_train_config(study: optuna.Trial, config: Config, input_dim: int) -> TrainConfig:
    """Return a train config for the tuning study."""
    if not config.tuning:
        return config.train

    train = copy.deepcopy(config.train)

    if len(config.train.model.hidden_layers) == 0:
        assert(config.tuning.hidden_layer_types), "hidden_layer_types must be specified."
        assert(config.tuning.hidden_layers_size_range), "hidden_layers_size_range must be specified."

        if config.tuning.split != None:
            assert(config.tuning.split_num_groups), "split_num_groups can't be empty"
            split = config.tuning.split
            l1, l1_sizes = suggest_layers(study, config, "pre_split", split[0])
            l2, l2_sizes = suggest_layers(study, config, "split", split[1])
            l3, _= suggest_layers(study, config, "post_split", split[2])
            hidden_layers = l1
            if len(l2) > 0:
                # Ensure input dim can be evenly split
                split_input_dim = l1_sizes[-1] if len(l1_sizes) > 0 else input_dim
                groups = study.suggest_categorical("spilt_num_groups", config.tuning.split_num_groups)
                if split_input_dim % groups != 0 :
                    good_dim = math.ceil(split_input_dim / groups) * groups
                    l1.append(f"linear({good_dim})")
                    l1_sizes.append(good_dim)

                group_dim = l1_sizes[-1] // groups
                join_dim = l2_sizes[-1] * groups
                hidden_layers = l1 + [f"split({group_dim})"] + l2 + [f"join({join_dim})"] + l3
        else:
            assert(config.tuning.n_hidden_layers), "n_hidden_layers must be specified."
            hidden_layers = suggest_layers(study, config, "hidden_layers", config.tuning.n_hidden_layers)

        if len(hidden_layers) == 0:
            hl_type = study.suggest_categorical(f"hidden_layer_type_last", config.tuning.hidden_layer_types)
            hidden_layers.append(hl_type)

        train.model.hidden_layers = hidden_layers

    if config.tuning.lr_range:
        train.optim.lr = str(study.suggest_float("lr", config.tuning.lr_range[0], config.tuning.lr_range[1], log=True))

    if config.tuning.nalu_lr_range:
        train.optim.nalu_lr = study.suggest_float("nalu_lr", config.tuning.nalu_lr_range[0], config.tuning.nalu_lr_range[1], log=True)

    if config.tuning.optimizer:
        train.optim.optimizer = str(study.suggest_categorical('optimizer', config.tuning.optimizer))

    if config.tuning.weight_decay_range:
        wd = config.tuning.weight_decay_range
        train.optim.weight_decay = study.suggest_float('weight_decay', wd[0], wd[1], log=True)

    train.batch_size = study.suggest_categorical("batch_size", config.tuning.batch_size_values)
    if config.tuning.tune_normalize:
        train.model.normalize = study.suggest_categorical('normalize', [True, False])
        train.model.normalizeX = train.model.normalize
        train.model.normalizeY = train.model.normalize

    return train

def prev_trails_count(study: optuna.Study):
    """
    Return number of unfailed trails so far.

    Useful for numbering the trails.
    """
    failed = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.FAIL:
            failed += 1

    good = len(study.trials) - failed
    return good

def read_env():
    config = {}
    with open(".env", "r") as f:
        for line in f.readlines():
            (key, val) = line.split("=")
            key = key.strip(" \n")
            val = val.strip(" \n")
            config[key] = val
    return config

def tune(config: Config, postgres: bool = False):
    """Run hyperparameter tuning based on given Config."""
    assert config.tuning, "Tuning config not provided."

    # Load or create study
    study_name = config.study_name
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    if postgres:
        env = read_env()
        user = env['PG_USER']
        password = env['PG_PASSWORD']
        host = env['PG_HOST']
        port = env['PG_PORT']
        db  = env["PG_DB"]
        storage = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        sqlite_file = config.logs_dir.joinpath("optuna.db")
        storage = f"sqlite:///{sqlite_file}"

    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=optuna.samplers.TPESampler(), storage=storage, load_if_exists=True)

    # Run the study
    dataset = load_dataset(config.dataset)
    def objective(trial: optuna.Trial) -> float:
        train_config = create_train_config(trial, config, dataset.input_dim())
        new_config = dataclasses.replace(config, train=train_config, tuning=None)
        trial.set_user_attr("config", pyrallis.dump(config))

        def pruning_callback(info):
            epoch = info['epoch']
            if config.train.evaluation_metric == 'val_loss':
                metric = info['val_loss']
            elif config.train.evaluation_metric == 'max_l1':
                metric = info['max_l1']
            else:
                raise ValueError(f"Invalid evauation_metric {config.train.evaluation_metric}")

            trial.report(metric, epoch)
            assert config.tuning

            if config.tuning.enable_prune:
                if trial.should_prune():
                    raise optuna.TrialPruned()

        trial_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(trial.number)
        return train_log(new_config, trial_name, callbacks=[pruning_callback], dataset=dataset)

    n_trials = config.tuning.trials - prev_trails_count(study)
    study.optimize(objective, n_trials=n_trials)

    # Save the study
    study_dir = config.logs_dir.joinpath(study_name)
    with open(study_dir.joinpath("optuna-study.pkl"), "wb") as f:
        pickle.dump(study, f)
