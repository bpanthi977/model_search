# Codebase Guide

## Setup

```bash
uv sync
source .venv/bin/activate
```

Uses `uv` for dependency management. Key dependencies: PyTorch, h5py, Optuna, pyrallis, seaborn.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point. Dispatches to train, validate, tune, or export. |
| `config.py` | All config dataclasses. Defines what YAML fields are valid. |
| `model.py` | MLP model builder. Defines all layer types. |
| `train.py` | Training loop, optimizer setup, checkpoint saving. |
| `dataset.py` | HDF5 dataset loader. |
| `tune.py` | Bayesian hyperparameter tuning via Optuna. |
| `grid_search.py` | Run training across a grid of hyperparameter combinations. |
| `visualize_activations.py` | Plot neuron activation distributions (violin plots). |
| `prune_model.py` | Remove dead neurons from a trained checkpoint. |
| `validate_model.py` | Evaluate a checkpoint; generate error histograms. |
| `lr_tune.py` | LR range test (sweep LR from 1e-6 to 10, plot loss). |
| `test_evd_model.py` | Detailed analysis of EVD models: Jacobians, sweep plots. |
| `visualize.py` | Plot loss curves and weight matrices. |
| `energy.py` | Visualize LULESH energy output differences. |
| `logs.py` | Utilities to read and compare multiple training run logs. |
| `utils.py` | Loss functions, hyperparameter extraction helpers. |

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

# Bayesian hyperparameter tuning
python main.py --tune --config configs/hgf_model_structure.yaml

# Grid search over hyperparameter combinations
python grid_search.py --config configs/hgf_model_structure.yaml --grid configs/hgf_model_structure_01.csv --wait --shared-memory

# Visualize neuron activations for a checkpoint
python visualize_activations.py --checkpoint logs/hgf-20p/20260101-120000-AbCd/ --dataset.subset -1

# Prune dead neurons from a checkpoint
python prune_model.py --checkpoint logs/hgf-20p/20260101-120000-AbCd/ --threshold 1e-9
```

## Config System

All behavior is controlled by YAML config files. Configs are parsed by
[pyrallis](https://github.com/eladrich/pyrallis) into typed Python dataclasses.

**Config hierarchy** (defined in `config.py`):
```
Config
├── dataset: DatasetConfig
│   ├── db_file         # Path to HDF5 dataset file
│   ├── label           # HDF5 group prefix (e.g. "EVD" → groups "EVD_train", "EVD_validate")
│   ├── subset          # Limit to first N samples (training only); -1 = all
│   └── sample          # Fraction of data to use (e.g. 0.2 = 20%)
│
├── train: TrainConfig
│   ├── epoch           # Number of training epochs
│   ├── batch_size
│   ├── device          # "cuda" or "cpu"
│   ├── loss            # "mse" or "smooth_l1"
│   ├── model: ModelConfig
│   │   ├── hidden_layers   # List of layer specs (see below)
│   │   ├── activation      # "relu", "tanh", "leaky_relu"
│   │   ├── normalize       # Shortcut: sets normalizeX and normalizeY
│   │   ├── normalizeX      # Z-score normalize inputs
│   │   └── normalizeY      # Z-score normalize outputs
│   └── optim: OptimizerConfig
│       ├── optimizer       # "adam", "adagrad", "rmsprop"
│       ├── lr              # Learning rate (string or float)
│       ├── weight_decay
│       └── nalu_lr         # Separate LR for NALU layers (optional)
│
├── tuning: TuningConfig    # Only needed for --tune mode
└── logs_dir                # Where to write run directories (default: "logs")
```

**Two-stage CLI parsing**: `argparse` handles top-level flags (`--checkpoint`, `--validate`, etc.),
and the remaining arguments are passed to pyrallis to override YAML fields. Example:
```bash
python main.py --config x.yaml --train.optim.lr 0.01 --train.epoch 300
```

## Hidden Layer Specification

The `hidden_layers` config field is a list of specs that define the network architecture. The
final output linear layer is added automatically.

| Spec | Effect |
|------|--------|
| `512` | Linear layer with 512 units |
| `"NALU(64)"` | Neural Arithmetic Logic Unit, 64 units |
| `"NALUi1(64)"` | NALU variant 1, 64 units |
| `"MULT0(64)"` | Multiplicative layer variant 0, 64 units |
| `"MULT1(64)"` | Multiplicative layer variant 1, 64 units |
| `"split(N)"` | Reshape `[batch, features]` → `[batch, N_groups, features/N_groups]`. Use N = number of groups. |
| `"join"` | Flatten `[batch, groups, ...]` → `[batch, groups × ...]` |
| `"any_spec-I"` | Append `-I` to skip activation after that layer |

**Wide / split-join networks** group input features and process each group through a shared layer
(like a 1D convolution). Specify N groups in `split(N)`, then the hidden size per group, then
`join` to merge. See `docs/research_overview.md` for details and the motivation.

Example config for a wide network:
```yaml
train:
  model:
    hidden_layers: ["split(3)", 24, "join", 200]
    activation: relu
    normalize: true
```
This creates: Split into 3 groups → shared linear 24 per group → join (72 total) → linear 200.

## Special Layer Types

**NALU (Neural Arithmetic Logic Unit)**: Learns exact arithmetic operations (+, -, ×, ÷) via
gating. Useful when the target function involves arithmetic. Needs a separate (often larger)
learning rate specified via `optim.nalu_lr`.

**MULT0 / MULT1**: Simpler multiplicative layers. Alternative to NALU for product-type operations.

**Split / Join**: No learnable parameters — just tensor reshaping. Used to implement wide networks.
`split(N)` takes the integer number of groups. The feature count must be divisible by N.

All layers use `torch.float64` (double precision) throughout the model.

## Dataset Format

Datasets are HDF5 files loaded by `dataset.py`. Structure:

```
dataset.h5
├── {label}_train      # Training data: shape [iterations, nodes, features]
└── {label}_validate   # Validation data: same shape
```

The raw 3D shape `[iterations, nodes, features]` is automatically reshaped to
`[iterations × nodes, features]` when loaded. The last `features/2` columns are treated as Y
(outputs) and the first half as X (inputs) — split is determined by the dataset shape as stored
by HPAC-ML.

The `label` in the config must match the HDF5 group prefix (e.g. `label: "EVD"` → reads
`EVD_train` and `EVD_validate`).

`dataset.subset` limits the number of training samples (rows after reshaping). `dataset.sample`
specifies a fraction (e.g. `0.2` = use 20% of data). When both are set, the smaller limit applies.

## Normalization

Input and output normalization (Z-score: subtract mean, divide by std) is applied **inside the
model's forward pass**, not as a preprocessing step. The normalization statistics (mean, std) are
computed from the training data once before training begins and stored inside the model. This means
the exported `model.pt` TorchScript includes normalization — you feed raw physical values in and
get raw physical values out.

Enable with `normalize: true` in the model config (or separately `normalizeX: true`,
`normalizeY: true`).

## Checkpoint System

Every training run creates a new directory:
```
logs/<study_name>/<YYYYMMDD-HHMMSS-RAND>/
├── config.yaml             # Full config used for this run
├── checkpoint.pth          # Latest checkpoint (epoch, model weights, optimizer state)
├── checkpoint_best.pth     # Best validation loss checkpoint
├── checkpoint-N.pth        # Periodic checkpoint (if --checkpoint-every-n N was used)
├── model.pt                # TorchScript export (runnable without Python model code)
├── train_loss.csv          # [epoch, loss, time, lr]
├── val_loss.csv            # [epoch, loss, time, max_l1]
├── info.csv                # [start_time, end_time, total_time, parameter_count]
├── model_shape             # Text summary of layer shapes
├── stdout.txt              # Redirected stdout (if -l flag used)
├── continue_from           # Path of checkpoint this run was resumed from
└── figs/                   # Plots from visualization scripts
```

**Resuming training**: When resuming from `--checkpoint`, the original directory is **copied** to
a new timestamped directory. Only `train.epoch`, `dataset.db_file`, and `study_name` may be
overridden when resuming (enforced in `main.py`). All other config fields are locked to the
original values.

The `Checkpoint` TypedDict in `config.py` contains: `epoch`, `best_val_loss`, `model_state_dict`,
`optimizer_state_dict`, `lr_scheduler_state_dict`, `continue_from`.

## Hyperparameter Tuning

**Bayesian tuning** (`--tune` mode, implemented in `tune.py`):
- Uses Optuna with Tree-Structured Parzen Estimator (TPE) and Median Pruning.
- Searches over: hidden layer count/sizes, activation, optimizer, LR, weight decay, normalization,
  split/join architecture configurations.
- Stores results in PostgreSQL (for multi-process parallel tuning) or local SQLite.
- Set up PostgreSQL credentials in a `.env` file: `PG_USER`, `PG_PASSWORD`, `PG_HOST`, `PG_PORT`,
  `PG_DB`.
- Launch N parallel tuning workers: `bash run_parallel_tune.sh N`

**Grid search** (`grid_search.py`):
- Define a CSV where each line is `--flag value1 value2 ...`. All combinations are run.
- `--shared-memory` shares the dataset tensor across processes (saves RAM).
- `--wait` blocks until all runs complete.

## Visualize Activations

`visualize_activations.py` captures post-activation outputs for each layer using PyTorch forward
hooks and plots them as violin plots.

```bash
python visualize_activations.py --checkpoint logs/run/ --dataset.subset -1
```

- `--dataset.subset -1` uses all data; set to a positive integer to limit to N batches.
- Output saved to `logs/run/figs/` as PNG files.
- Also saves a CSV per layer with per-neuron mean/min/max/std statistics.
- Hooks are placed on activation functions (ReLU, tanh, etc.) rather than linear layers, so the
  plots show post-activation values — directly indicating dead neurons.
- Wide layers (>32 neurons) are split into rows of 32 for readability.
- Outlier filtering: values above the 95th percentile per neuron are masked (configurable via
  `--percentile`). Filtering is per-neuron to avoid discarding valid data for other neurons.

## Pruning Dead Neurons

`prune_model.py` removes neurons whose max absolute activation across the dataset is below a
threshold.

```bash
python prune_model.py --checkpoint logs/run/ --threshold 1e-9
```

**Process:**
1. Run a forward pass on the validation set.
2. Track `max(|activation|)` per neuron across all batches.
3. For each layer, keep only neurons where `max_abs > threshold`.
4. Slice weight matrices and biases to remove dead rows/columns.
5. Save a new checkpoint and config to `<checkpoint_dir>_pruned/`.
6. Print a summary and verify the pruned model produces the same outputs as the original.

**Layer-specific handling:**
- `Linear`: Remove rows (output dim) and corresponding columns in the next layer (input dim).
- `BatchNorm1d`: Subset weight, bias, running_mean, running_var.
- `NALU / MULT`: Subset `W_hat`, `M_hat`, `G` parameters accordingly.
- `Split → Linear → Join`: Indices are replicated across groups for the join case. If the index
  mapping doesn't work cleanly (e.g. between a split and a non-grouped layer), all inputs are kept.

After pruning, optionally fine-tune for a few epochs to recover any marginal accuracy loss.
