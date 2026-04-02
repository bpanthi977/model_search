# Design Decisions: visualize_activations.py

## Purpose
This file serves as a summarized memory log for future coding agents working on `visualize_activations.py`. It documents the architectural and design choices made during the script's creation to ensure future modifications respect the original constraints and goals.

## Architecture & Implementation Choices

1. **Non-Intrusive Activation Capture (Forward Hooks)**
   - **Decision**: PyTorch `register_forward_hook` is used exclusively on "leaf" modules (modules with no children, excluding `nn.Dropout`, `nn.BatchNorm`, `Split`, `SplitInterleave`, and `Join`). Additionally, computation layers (e.g., `nn.Linear`, `NALU`, `MULT`) that are immediately followed by an activation function (within 1-2 positions, accounting for `BatchNorm`) are skipped because their post-activation output is more informative for pruning.
   - **Reasoning**: This allows the script to capture intermediate activations gracefully without requiring any modifications to the underlying model class definitions. `Split`, `SplitInterleave`, and `Join` are excluded because they are parameterless reshape layers that carry no pruning information and would only add noise to the visualization. Computation layers before activations are skipped because a neuron's ReLU output is more informative than its pre-activation linear output: a dead ReLU (always 0) indicates a neuron that can be pruned, whereas the pre-activation output is redundant. This reduces plot count and focuses visualization on the most actionable information.

2. **Plotting Library (Seaborn Violin Plots)**
   - **Decision**: Seaborn's `violinplot` was explicitly chosen over boxplots or histograms.
   - **Reasoning**: Violin plots provide a much clearer view of the full distributional density of activations across a batch, which is crucial for identifying dead neurons or unusual multi-modal activation patterns.

3. **Handling Wide Layers (Subplot Grids)**
   - **Decision**: Layers with more than 32 neurons are split into a vertical grid of subplots, with a maximum of 32 neurons per row.
   - **Reasoning**: Without splitting, architectures with wide layers (e.g., 512 neurons) would result in severely squished, unreadable x-axes. Wrapping at 32 neurons ensures each violin plot remains distinct and legible.

4. **Y-Axis Consistency Across Grids**
   - **Decision**: `plt.subplots(..., sharey=True)` is strictly enforced for multi-row plots.
   - **Reasoning**: This mathematically locks the vertical scale across all rows for a given layer. Without this, Seaborn would auto-scale each row independently, destroying the ability to visually compare the magnitude of activations between a neuron on row 1 and a neuron on row 4.

5. **CSV Summary Statistics (Per-Neuron)**
   - **Decision**: For each layer plot, a CSV file is saved alongside the PNG with the same base name (e.g., `model_2.png` → `model_2.csv`). The CSV has one row per neuron with columns: `neuron`, `mean`, `min`, `max`, `std`.
   - **Reasoning**: Statistics are computed from the **raw, unfiltered** activations (before percentile masking) so that min/max reflect the true activation range rather than the truncated visualization window. This provides a machine-readable companion to the violin plots, useful for downstream analysis (e.g., identifying dead neurons programmatically).

6. **Outlier Filtering (Per-Neuron Percentiles)**
   - **Decision**: Outlier filtering (controlled by `--percentile`, default 95.0) is calculated **per neuron**, not globally across the layer. Out-of-bounds values are masked with `np.nan`.
   - **Reasoning**: 
     - Calculating per-neuron prevents high-variance neurons from skewing the cutoffs for stable neurons.
     - Masking with `NaN` (which Seaborn automatically ignores) is critical. Dropping the entire sample row from the DataFrame because *one* neuron had an outlier would unfairly delete perfectly valid activation data for the other 511 neurons in that batch.

6. **Configuration Overrides (Pyrallis `args_rest`)**
   - **Decision**: The script intentionally parses only its own arguments (`--checkpoint`, `--percentile`) using `argparse.parse_known_args()`, passing the rest directly to `pyrallis`.
   - **Reasoning**: This allows users to dynamically override any deeply nested config parameter (e.g., `--train.batch_size 1024`, `--dataset.sample 0.5`) from the command line without hardcoding those specific arguments into the script. The script also specifically repurposes `--dataset.subset` to represent the number of inference batches to process (`-1` means all data).
