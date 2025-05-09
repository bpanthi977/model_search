import os
import pickle
import pyrallis
import hashlib
import torch
import optuna
import sys
import dataclasses

from dataset import load_dataset
from model import train, evaluate_model, create_model
from config import Config, TrainConfig

def run_name(config: Config):
    config_str = pyrallis.dump(config)
    config_hash = hashlib.sha1(config_str.encode()).hexdigest()
    return config_hash


def create_train_config(study: optuna.Study, config: Config) -> TrainConfig:
    n_hidden_layers = study.suggest_categorical("hidden_layers", config.tuning.n_hidden_layers)
    if n_hidden_layers > 0:
        hidden_layer_size = study.suggest_categorical("hidden_layers_sizes", config.tuning.hidden_layers_sizes)
        hidden_layers = [hidden_layer_size] * n_hidden_layers
    else:
        hidden_layers = []

    train = dataclasses.replace(config.train) # clone
    train.hidden_layers = hidden_layers
    train.lr = study.suggest_categorical("lr", config.tuning.lr_values)
    train.batch_size = study.suggest_categorical("batch_size", config.tuning.batch_size_values)

    return train


def train_eval(config: Config):
    # Load data, train and evaluate
    dataset = load_dataset(config.dataset)
    model = create_model(config.train, dataset)
    train(model, dataset, config.train)
    loss = evaluate_model(model, dataset, config.train.batch_size)

    # Save config and model
    run_dir = config.logs_dir.joinpath(config.study_name).joinpath(run_name(config))
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir.joinpath("config.yaml"), "w") as f:
        f.write(pyrallis.dump(config))

    model_script = torch.jit.script(model.to(torch.device('cpu')).double())
    model_script.save(run_dir.joinpath("model.pt"))

    # Return the evaluation metric
    return loss


if __name__ == "__main__":
    config = pyrallis.parse(config_class=Config)

    # Load or create study
    study_name = config.study_name
    sqlite_file = config.logs_dir.joinpath(study_name).joinpath("optuna.db")
    storage = f"sqlite:///{sqlite_file}"
    if not os.path.exists(sqlite_file):
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage)
    else:
        study = optuna.load_study(study_name=study_name, storage=storage)

    # Run the study
    def objective(trial: optuna.Study):
        train_config = create_train_config(trial, config)
        new_config = dataclasses.replace(config, train=train_config)
        return train_eval(new_config)


    study.optimize(objective, n_trials=config.tuning.trials)

    # Save the study
    with open(study_dir.joinpath("study.pkl"), "wb") as f:
        pickle.dump(study, f)
