"""Hyperparameter tuning."""
import pickle
import dataclasses
import optuna
from pathlib import Path

from train import train_log
from config import Config, TrainConfig


def create_train_config(study: optuna.Trial, config: Config) -> TrainConfig:
    """Return a train config for the tuning study."""
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

def tune(config: Config):
    """Run hyperparameter tuning based on given Config."""
    assert config.tuning, "Tuning config not provided."

    # Load or create study
    study_name = config.study_name
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    sqlite_file = config.logs_dir.joinpath("optuna.db")
    storage = f"sqlite:///{sqlite_file}"
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=optuna.samplers.TPESampler(), storage=storage, load_if_exists=True)

    # Run the study
    def objective(trial: optuna.Trial) -> float:
        train_config = create_train_config(trial, config)
        new_config = dataclasses.replace(config, train=train_config)

        def pruning_callback(epoch, loss):
            trial.report(loss, epoch)
            assert config.tuning

            if config.tuning.enable_prune:
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return train_log(new_config, trial.number, callbacks=[pruning_callback])

    n_trials = config.tuning.trials - prev_trails_count(study)
    study.optimize(objective, n_trials=n_trials)

    # Save the study
    study_dir = config.logs_dir.joinpath(study_name)
    with open(study_dir.joinpath("optuna-study.pkl"), "wb") as f:
        pickle.dump(study, f)
