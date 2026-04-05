# Workflow: Training Surrogate Models for HPC Functions

This document walks through the end-to-end process used in this research to find and evaluate
neural network surrogate models for functions inside an HPC application (LULESH).

## Step 1: Profile the Application

Use Intel VTune (or similar) to identify hotspot functions:

1. Compile the application with function inlining disabled.
2. Run a CPU Hotspot profile.
3. Identify computationally expensive functions that:
   - Take well-defined numeric inputs and produce numeric outputs.
   - Are called many times per simulation (not just once).
   - Don't just copy data — they compute something non-trivial.

In this work, `CalcHourglassControlForElems._omp_fn.0` was the biggest hotspot but only copies
data, so it was skipped. The three chosen functions (EVD, HGF, VFE) all do computation.

## Step 2: Annotate Code and Collect Data

Use **HPAC-ML** (`#pragma approx ml`) to annotate the code block containing the function call.
This requires manually identifying which arrays are inputs and which are outputs.

Example annotation for EVD:
```c
#pragma approx declare tensor_functor(i3_map: [i,k] = ([i,j],[i,j],[i,j]))
#pragma approx declare tensor_functor(i1_map: [i,j] = ([i,j]))
#pragma approx declare tensor(input: i3_map(x8n[0:numElem][0:8], y8n[0:numElem][0:8], z8n[0:numElem][0:8]))
#pragma approx ml(offline) in(input) out(i1_map(out[0:numElem][0:24])) label("EVD")
{
  // loop that calls CalcElemVolumeDerivative for each element
}
```

Run the annotated program with the `HPAC_DB_FILE` environment variable set to write data to an
HDF5 file. Each call to the annotated region appends a row to `{label}_train` group in the file.

**Sanity check**: Run the program multiple times and compare statistics (mean, std, range) across
the resulting datasets. Mismatches indicate annotation errors (e.g. wrong array, wrong size).

**Data size note**: For large functions, data collection can produce very large files (EVD: ~9 GB
for full dataset, VFE: >80 GB at 100%). Collect a subset if needed (e.g. every 10th iteration).

## Step 3: Initial Hyperparameter Search

Start with a small dataset (1–20% of full data) and few epochs (20). Run many models in parallel
to explore the search space quickly.

### Set up the config

```yaml
# configs/evd_search.yaml
dataset:
  db_file: dataset/EVD-20p.h5
  label: EVD
  sample: 1.0           # use all of this file (which is 20% of full data)
study_name: evd-20p-search
train:
  epoch: 20
  batch_size: 1024
  device: cuda
  loss: mse
  model:
    normalize: true
    hidden_layers: []   # will be overridden by tuner
  optim:
    optimizer: adagrad
    lr: "0.01"
tuning:
  trials: 100
  n_hidden_layers: [2, 3, 4]
  hidden_layers_size_range: [24, 512]
  split_num_groups: [1, 3, 9]
  enable_prune: true
```

### Run Bayesian tuning

```bash
# Single process
python main.py --tune --config configs/evd_search.yaml

# Multiple parallel workers (uses PostgreSQL)
bash run_parallel_tune.sh 4
```

Monitor with the Optuna dashboard:
```bash
bash start_dashboard.sh
# opens http://localhost:8080
```

### Check early if training is on the right track

Look at the validation loss curve for the best models. If the curve has **flattened from the very
first epochs**, the dataset is too small — you need more data before tuning is useful. This was
observed for HGF on 1% dataset and VFE.

## Step 4: Scale Up Training

Take the top 5–10 models from Step 3. Retrain them on a larger dataset (20–100%) for more epochs
(100–300).

```bash
# Resume from a promising checkpoint with new dataset and more epochs
python main.py \
  --checkpoint logs/evd-20p-search/20260101-120000-AbCd/ \
  --dataset.db_file dataset/EVD-100p.h5 \
  --train.epoch 100
```

Only `train.epoch`, `dataset.db_file`, and `study_name` can be changed on resume. All other
config (model architecture, optimizer, etc.) is locked to the original checkpoint.

**Budget management**: Training one large model can take hours. Use the validation loss trend to
decide whether to continue. A still-decreasing curve → continue. A flattened curve → stop.

## Step 5: Visualize Activations

After training, check whether neurons are actually being used:

```bash
python visualize_activations.py \
  --checkpoint logs/evd-20p-search/20260101-120000-AbCd/ \
  --dataset.subset -1
```

Output is saved to `logs/.../figs/`. Look for neurons whose violin plots are a flat line near
zero — those are dead neurons candidates for pruning. The companion CSV files contain per-neuron
min/max/std for programmatic analysis.

## Step 6: Prune Dead Neurons

```bash
python prune_model.py \
  --checkpoint logs/evd-20p-search/20260101-120000-AbCd/ \
  --threshold 1e-9
```

The script prints a layer-by-layer reduction summary and verifies the pruned model produces
identical outputs to the original. The pruned checkpoint is saved to a `_pruned` subdirectory.

Typical results: EVD model shrank 95% (20k → 949 parameters) with no accuracy loss. HGF model
shrank 50% with minor accuracy drop.

If accuracy drops noticeably, fine-tune the pruned model for a few epochs:
```bash
python main.py --checkpoint logs/evd-20p-search/20260101-120000-AbCd_pruned/ --train.epoch 10
```

## Step 7: Export TorchScript Model

```bash
python main.py --create-model --checkpoint logs/evd-20p-search/20260101-120000-AbCd/
```

This creates `model.pt` in the checkpoint directory — a self-contained TorchScript file that
includes the normalization layers. LULESH's HPAC-ML inference uses this file.

## Step 8: Evaluate in the Application

Change the HPAC-ML annotation from `ml(offline)` to `ml(infer)` and provide the model path:

```c
#pragma approx ml(infer) in(input) out(i1_map(out[0:numElem][0:24])) \
  model("/path/to/model.pt") label("EVD")
```

Or use the `SURROGATE_MODEL` environment variable:
```bash
SURROGATE_MODEL=/path/to/model.pt ./lulesh2.0 -s 30
```

**Accuracy metric**: LULESH reports a symmetry check at the end, but this is a weak measure.
The metric used in this research is the **mean absolute error of the final energy values** compared
to the unmodified program run. Threshold used: Energy MAE < 2.0 (mean energy ~239).

## Step 9: Test on Different Domain Sizes

If the model was trained on grid size 30×30×30, test on 40×40×40:
```bash
SURROGATE_MODEL=/path/to/model.pt ./lulesh2.0 -s 40
```

If accuracy is not acceptable, **retrain from the existing checkpoint** on data from the new
domain size — this is much more efficient than training from scratch. Just 10 extra epochs on
the new-size dataset can beat 60 epochs of training from scratch.

## Tips From This Research

**Feature ordering matters for split/join networks.**
For EVD's `[x0..x7, y0..y7, z0..z7]` ordering, split(3) groups x, y, z naturally. Reordering
to `[x0,y0,z0, ...]` causes the same model to get 100× worse validation loss. When using split
networks, make sure the input tensor is arranged so that related features are contiguous.

**Flat validation curve early = need more data.**
If the validation loss doesn't decrease even in the first few epochs, increasing epochs won't
help. You need a larger dataset. This was the blocker for VFE (would need ~80+ GB).

**Validation loss ≠ application accuracy.**
Always test the model in the actual application. In some cases a model with *higher* validation
loss gave *lower* energy error in LULESH. Use the application-specific metric as the final judge.

**Wide models can outperform deep models at smaller size.**
For EVD, a wide `3x24-200` model (20k params) achieved lower energy error than a larger `160-485`
model (93k params). The split/join structure leverages domain structure efficiently.

**AdaGrad with lr=0.01 was consistently good for these problems.**
Adam worked for EVD, but AdaGrad showed more stable convergence across a wider range of learning
rates in ablation studies. ReLU activation was better than tanh. Batch normalization increased
loss and was not used.

**Multiple models can be used simultaneously.**
If two functions are not on the same call stack (e.g. EVD and HGF), models for both can run at
the same time in the application. The combined speedup/accuracy is roughly additive but should
be tested empirically.
