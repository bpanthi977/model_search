# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Uses `uv` for dependency management:
```bash
uv sync
source .venv/bin/activate
```

## Common Commands

```bash
# Train from a config file
python main.py --config configs/hgf_model_structure.yaml

# Override config fields on the command line
python main.py --config configs/hgf_model_structure.yaml --train.model.activation tanh --train.epoch 200

# Resume training from a checkpoint
python main.py --checkpoint logs/hgf-20p/20260101-120000-AbCd/

# Validate a checkpoint on the dataset
python main.py --validate --checkpoint logs/hgf-20p/20260101-120000-AbCd/

# Export a TorchScript model.pt from a checkpoint
python main.py --create-model --checkpoint logs/hgf-20p/20260101-120000-AbCd/

# Run grid search over hyperparameter combinations
python grid_search.py --config configs/hgf_model_structure.yaml --grid configs/hgf_model_structure_01.csv --wait --shared-memory

# Visualize neuron activations for a checkpoint
python visualize_activations.py --checkpoint logs/hgf-20p/20260101-120000-AbCd/ --dataset.subset -1

# Bayesian hyperparameter tuning
python main.py --tune --config configs/hgf_model_structure.yaml
```

## Architecture Overview

### Config System
All behavior is driven by YAML config files parsed with **pyrallis** into typed dataclasses in `config.py`. The top-level `Config` has four nested sections: `dataset` (`DatasetConfig`), `train` (`TrainConfig` → contains `model: ModelConfig` and `optim: OptimizerConfig`), `tuning` (`TuningConfig`, optional), and `logs_dir`.

**Two-stage argument parsing** is used throughout: `argparse` handles top-level flags (`--checkpoint`, `--validate`, etc.), and the remaining `args_rest` list is passed directly to `pyrallis.parse()` to override YAML fields (e.g. `--train.model.hidden_layers [512,512]`).

### Model Architecture (`model.py`)
`MLP` is an `nn.Sequential`-based model built from `config.model.hidden_layers`, a list of `Union[int, str]` specs:
- Integer → `nn.Linear`
- `"NALU(64)"`, `"NALUi1(64)"` → Neural Arithmetic Logic Unit variants (for learning arithmetic operations)
- `"MULT0(64)"`, `"MULT1(64)"` → multiplication layers
- `"split(2)"` / `"join"` → reshape layers (no learnable parameters)
- Append `-I` to any spec (e.g. `"NALU(64)-I"`) to skip the activation after that layer

All weights use `torch.float64`. Input/output normalization (mean/std from training data) is applied inside the `MLP.forward()` method, not as a preprocessing step. The final output linear layer is added automatically — `hidden_layers` specifies only the hidden layers.

`create_model(config: TrainConfig, dataset: Dataset)` is defined in `model.py`

### Training & Checkpoints (`train.py`)
`train_log()` creates the run directory at `logs/<study_name>/<YYYYMMDD-HHMMSS-XXXX>/`, saves `config.yaml` there, runs the training loop, and saves:
- `checkpoint.pth` — latest checkpoint
- `checkpoint_best.pth` — best validation loss checkpoint
- `model.pt` — TorchScript export

The `Checkpoint` TypedDict (in `config.py`) holds: `epoch`, `best_val_loss`, `model_state_dict`, `optimizer_state_dict`, `lr_scheduler_state_dict`, `continue_from`.

When resuming from `--checkpoint`, the original run directory is **copied** to a new timestamped directory. Only `train.epoch`, `dataset.db_file`, and `study_name` are allowed to be overridden when resuming (enforced in `main.py`).

### Dataset (`dataset.py`)
Reads HDF5 files (h5py). Expects groups named `{label}_train` and `{label}_validate`. The raw shape is `[iterations, nodes, features]`, reshaped to `[iterations*nodes, features]`. `dataset.subset` in the top-level `DatasetConfig` limits the number of *samples* (rows); in visualization scripts, `--dataset.subset` is repurposed to mean number of *batches*.

### Logs Directory Layout
```
logs/
  <study_name>/
	<YYYYMMDD-HHMMSS-RAND>/
	  config.yaml
	  checkpoint.pth
	  checkpoint_best.pth
	  checkpoint-N.pth    # if --checkpoint-every-n N was used
	  model.pt
	  stdout.txt          # if -l flag was used
	  continue_from       # if resumed from another checkpoint
	  figs/               # visualizations
```

### Design Documentation files

For some files there is documentation on the design decisions
taken. When editing those files, read the respective
documentation. Update the documentation file if the design is
changed. The files are in docs/ folder:
- prune_model_plan.md
- test_evd_model_design.md
- visualize_activations_design.md
