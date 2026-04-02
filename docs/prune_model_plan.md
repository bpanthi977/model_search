# Plan: prune_model.py

## Objective
Implement a structured pruning script (`prune_model.py`) that identifies inactive neurons and generates a new, smaller model with reduced layer dimensions.

## Background
From the research log (2026-03-18):
- Visualizing activations revealed many neurons don't activate (remain zero)
- Hypothesis: Prune those inactive layers to create a more efficient model

## Implementation Plan

### 1. Configuration & Loading
- **Inputs:**
  - `--checkpoint`: Path to checkpoint directory containing `checkpoint.pth` and `config.yaml`
  - `--threshold`: Activation threshold below which a neuron is "inactive" (Default: `1e-9`)
  - `--dataset.subset`: Number of batches for activation analysis (Default: `-1` for all data)
  - `--output_dir`: Where to save pruned model (Default: `<checkpoint>_pruned`)
- Load config using `pyrallis`, dataset using `load_dataset`, model using `create_model`

### 2. Activation Analysis Phase
- For hidden layers followed by an activation function (ReLU, Tanh, LeakyReLU), register the hook on the **activation layer** to measure post-activation output. This correctly identifies dead neurons (neurons that output 0 under ReLU even if their linear pre-activation is large and negative).
- For layers with no following activation (e.g. NALU-I, final output layer), register the hook on the **prunable layer** itself to measure pre-activation output.
- Run inference on validation dataset
- Track `max(abs(activation))` per neuron across all batches
- Identify "keep indices": neurons where `max_abs > threshold`

### 3. Structured Pruning Logic
- **Linear Layers:**
  - Remove rows from weight matrix and bias using current layer's keep indices
  - Remove columns from next layer's weight matrix using next layer's keep indices
- **BatchNorm1d:**
  - Subset weight, bias, running_mean, running_var using current layer's keep indices
- **Custom Layers (NALU, MULT):**
  - Apply similar subsetting to `W_hat`, `M_hat`, `G` parameters

### 4. Verification & Output
- Print summary: Layer | Original | Pruned | % Reduction
- Run parity check: verify original and pruned models produce identical outputs
- Save pruned `config.yaml` and `checkpoint.pth` to output directory

## Technical Notes
- Uses `float64` (as per model dtype)
- Threshold default `1e-9` to catch effectively zero neurons
- Target: `nn.Sequential` MLP architectures

### Split/Join layer handling in keep_input
When computing `keep_input` for a prunable layer, the indices propagated from the previous prunable layer must account for intermediate reshape layers:
- **Join between two prunable layers**: `curr_in` will be a multiple of `prev_out`. Indices are replicated across groups: `ki = cat([ki + g*prev_out for g in range(n_groups)])`.
- **Split between two prunable layers** (or any other dimension mismatch): `curr_in != prev_out` and the join rule doesn't apply. In this case `keep_input` is set to `None` (all input columns kept) because the keep-output indices of the previous layer cannot be mapped into the current layer's input space. This avoids an out-of-bounds CUDA index error.
