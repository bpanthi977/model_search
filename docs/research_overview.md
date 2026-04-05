# Research Overview

## What This Work Is About

This repository supports an MSc thesis studying the use of **local surrogate models** to accelerate
HPC (High Performance Computing) applications. The central idea: instead of replacing an entire
simulation with a neural network (a "global" surrogate), we replace small, frequently-called
functions inside the application with small neural networks. These are called **local surrogate
models**.

The tradeoff being studied:
- **Global surrogate**: replaces the whole application. Big model, big data, hard to generalize to
  different problem sizes.
- **Local surrogate**: replaces one small function called thousands of times per simulation. Smaller
  model, smaller data, better chance of generalizing across problem sizes.

The thesis provides empirical evidence of where local surrogates work well, where they struggle,
and what determines success.

## Testbed: LULESH

LULESH (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics) is a proxy HPC application
simulating shock hydrodynamics on a 3D hexahedral mesh. It runs ~1000 iterations, each updating
node positions, velocities, forces, and element properties.

Three functions were chosen as surrogate targets (profiled using Intel VTune):

| Nickname | Full Function Name | Inputs | Outputs | % of Runtime |
|----------|--------------------|--------|---------|--------------|
| **EVD** | `CalcElemVolumeDerivative` | 24 floats (x,y,z coords of 8 nodes) | 24 floats (volume derivatives) | ~8.9% |
| **HGF** | `CalcFBHourglassForceForElems._omp_fn.0` | 74 floats (coords, velocities, vol. derivatives) | 24 floats (hourglass forces) | ~13.2% |
| **VFE** | `CalcVolumeForceForElems` | 439 floats | 3 floats | ~59% |

EVD is the simplest (leaf-level, small input/output), VFE is the most complex (encompasses HGF and
other computations).

## Key Findings

**EVD (small function — success)**
- A wide model with hidden layers `3x24-200` (20k parameters) achieved Energy MAE of 0.42 — well
  within acceptable error (threshold: 2.0).
- This model runs LULESH in **15 seconds vs 25 seconds** baseline — a **40% speedup**.
- The model generalizes to a larger domain size (40×40×40) with Energy MAE of 1.52 — still
  acceptable (trained on 30×30×30).
- After pruning dead neurons, the model shrank from 20k to 949 parameters with no increase in
  error.

**HGF (medium function — partial success)**
- Required ~389k parameters to achieve acceptable accuracy (Energy MAE 1.975 < 2.0).
- Execution time increased to 115 seconds (4.6× slower than baseline).
- Training on 20% of data took 72 hours; full dataset training took ~24 hours/model on V100 GPUs.
- Model does not generalize to larger domain sizes.

**VFE (large function — failure)**
- 439 inputs made training difficult. Even 2M-parameter models plateaued at validation loss ~0.085
  (normalized), and the program produced incorrect results when using the model.
- The validation loss curve flattened early, suggesting more data is needed but was not feasible
  to collect.

## Wide / Split-Join Networks

A key architectural finding: **wide networks** that exploit repeated structure in the input work
well. For inputs like EVD's `[x0..x7, y0..y7, z0..z7]`, the first layer is split into 3 groups
(one per coordinate axis), each processed identically by a shared layer. This is represented in
config as `"split(3)"` followed by a hidden layer size, then `"join"`.

Example: `hidden_layers: ["split(3)", 24, "join", 200]` means:
1. Split input into 3 groups
2. Apply a shared linear layer to each group → 24 outputs each
3. Join back → 72 features
4. Apply a linear layer → 200 features

Benefits: fewer parameters, better generalization, faster training. Domain knowledge about
repeated structure (e.g., x, y, z coordinates follow the same formula) guides the grouping.

**Feature ordering matters**: Grouping only works if related features are contiguous. `[x0..x7,
y0..y7, z0..z7]` allows split(3); `[x0,y0,z0, x1,y1,z1, ...]` breaks it. Incorrect ordering
can increase validation loss by 100×.

## Model Pruning

After training, visualizing neuron activations often reveals neurons that never activate (output
stays near zero for the entire dataset). These can be removed with no effect on accuracy.

For ReLU networks: prune neurons where `max(|activation|) < 1e-9` across the dataset.

EVD model `3x24-200` pruned from 20k → 949 parameters (95% reduction) with the same Energy MAE.
HGF model `303-3x264-256-485` pruned from 389k → 194k parameters (50% reduction) with slight
accuracy drop (Energy MAE 1.975 → 2.44, still near threshold).

## Retraining Across Domain Sizes

When a model trained on grid 30×30×30 is tested on grid 40×40×40 and fails, continuing training
from that checkpoint on new data is far more efficient than training from scratch:
- **Retrained model (10 extra epochs)** outperforms **scratch model (60 epochs)** on the new grid.
- The retrained model also maintains better accuracy on the original grid size.

## Workflow Summary

1. **Profile** application (VTune) to find hotspot functions
2. **Annotate** code with HPAC-ML `#pragma approx ml` to collect input/output data
3. **Explore** data: check statistics across multiple runs for annotation errors
4. **Search**: train many models on small dataset (1-20%) with Bayesian hyperparameter tuning
5. **Scale up**: take top models, train on larger dataset
6. **Visualize activations** to check for dead neurons
7. **Prune** dead neurons; fine-tune if needed
8. **Evaluate** by replacing function in LULESH and measuring final energy error

For detailed methodology, see the full thesis.
