import os
import pickle
import pyrallis
import torch
import optuna
import sys
import dataclasses

from dataset import load_dataset
from model import train, evaluate_model, create_model
from config import Config, TrainConfig

def create_train_config(study: optuna.Study, config: Config) -> TrainConfig:
    if not config.train.hidden_layers:
        n_hidden_layers = study.suggest_categorical("hidden_layers", config.tuning.n_hidden_layers)
        if n_hidden_layers > 0:
            hidden_layer_size = study.suggest_categorical("hidden_layers_sizes", config.tuning.hidden_layers_sizes)
            hidden_layers = [hidden_layer_size] * n_hidden_layers
        else:
            hidden_layers = []

    train = dataclasses.replace(config.train) # clone
    train.hidden_layers = config.train.hidden_layers or hidden_layers
    train.lr = config.train.lr or study.suggest_categorical("lr", config.tuning.lr_values)
    train.batch_size = config.train.batch_size or study.suggest_categorical("batch_size", config.tuning.batch_size_values)

    return train


def train_eval(config: Config, trial_id: int, callbacks):
    # Load data, train and evaluate
    dataset = load_dataset(config.dataset)
    model = create_model(config.train, dataset)
    train(model, dataset, config.train, callbacks)
    loss = evaluate_model(model, dataset, config.train.batch_size)

    # Save config and model
    run_dir = config.logs_dir.joinpath(config.study_name).joinpath(f"{trial_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir.joinpath("config.yaml"), "w") as f:
        f.write(pyrallis.dump(config))

    model_script = torch.jit.script(model.to(torch.device('cpu')).double())
    model_script.save(run_dir.joinpath("model.pt"))

    # Return the evaluation metric
    return loss

def prev_trails_count(study: optuna.Study):
    failed = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.FAIL:
            failed += 1

    good = len(study.trials) - failed
    return good

if __name__ == "__main__":
    config = pyrallis.parse(config_class=Config)

    # Load or create study
    study_name = config.study_name
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    sqlite_file = config.logs_dir.joinpath("optuna.db")
    storage = f"sqlite:///{sqlite_file}"
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=optuna.samplers.TPESampler(), storage=storage, load_if_exists=True)

    # Run the study
    def objective(trial: optuna.Study):
        train_config = create_train_config(trial, config)
        new_config = dataclasses.replace(config, train=train_config)

        def pruning_callback(epoch, loss):
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return train_eval(new_config, trial.number, callbacks=[pruning_callback])


    n_trials = config.tuning.trials - prev_trails_count(study)
    study.optimize(objective, n_trials=n_trials)

    # Save the study
    study_dir = config.logs_dir.joinpath(study_name)
    with open(study_dir.joinpath("optuna-study.pkl"), "wb") as f:
        pickle.dump(study, f)
