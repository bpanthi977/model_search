# Design Decisions: `test_evd_model.py`

This document serves as a memory for coding agents working on or modifying `test_evd_model.py`. It summarizes the architectural decisions and trade-offs made during its creation.

## 1. Native Python Porting over C++ Binding
**Decision**: The LULESH `CalcElemVolumeDerivative` and `VoluDer` functions were ported natively to batched, vectorized NumPy Python functions.
**Reasoning**: Avoids complex C++ bindings (like pybind11 or ctypes). Vectorization allows for lightning-fast batch evaluation during parameter sweeps and finite-difference Jacobian calculations.

## 2. Model Loading Strategy (TorchScript)
**Decision**: The script loads exported `.pt` TorchScript models (`torch.jit.load`) rather than reconstructing the model via `config.yaml` and loading `checkpoint.pth`.
**Reasoning**: The `Normalization` module (which applies Z-score normalization to inputs and denormalizes outputs) is baked directly into the `.forward()` method of the TorchScript. Loading the `.pt` file means the script can feed raw, unnormalized physical coordinates directly into the model and compare the unnormalized output 1:1 with the physical mathematical functions.

## 3. Handling Device Mismatches
**Decision**: The script explicitly checks for CUDA and moves both the input tensor and the model to the target device, then extracts back to the CPU.
**Reasoning**: During development, moving a TorchScript model to CPU (`model.cpu()`) failed because the normalization constants were packed in standard Python tuples `(bool, Tensor, Tensor)` inside the script, causing them to stay on the GPU while the model layers moved to the CPU. Enforcing device parity at inference avoids this edge case.

## 4. Visualization Strategy (The 24x24 Problem)
**Decision**: Visualizing 24 inputs to 24 outputs requires 576 relationships. We adopted a multi-tiered approach:
*   **Jacobian Heatmaps (Idea 1)**: Computes finite-difference Jacobians at a specific baseline point. This reveals the "structural zeros" dictated by physics (e.g., node 0's X coordinate shouldn't affect node 5's Y derivative) and instantly visualizes whether the NN learned the correct physical topology.
*   **Global Max Error Matrix (Idea 2)**: Sweeps every input and compresses the errors into a 24x24 heatmap, revealing worst-case performance domains.
*   **6x4 Grid Sweep Plots**: When varying specific inputs (`--vary-in`), it creates a 24-plot grid (6 rows, 4 columns) showing the exact True vs Model curves.

## 5. Physical Feature Naming
**Decision**: Instead of mapping features as generic arrays `X[0...23]`, we mapped them back to the physics indices `x[0..7]`, `y[0..7]`, `z[0..7]` and `dvdx, dvdy, dvdz`.
**Reasoning**: Ensures plots are readable by domain scientists.

## 6. Baseline Data Sampling and Context
**Decision**: Constant features (those not being swept) are sampled from actual training data (e.g., `EVD-20p.h5`) rather than random uniform space.
**Reasoning**: Evaluates the model within its expected in-distribution manifold. The 23 constant values are cleanly printed in a 3-column layout alongside the 6x4 grids to ensure complete reproducibility of the plot.

## 7. Reproducibility Scripts
**Decision**: The script auto-generates an executable `cmd.sh` script inside the output directory.
**Reasoning**: Saves the exact CLI invocation (dataset used, input varied, out-dir) alongside the generated plots, so anyone (or any agent) reviewing the plots later knows exactly how to regenerate them without scrolling through shell history.
