# Documentation Index

This repository is a PyTorch-based training and hyperparameter search framework built to study
neural network surrogate models for HPC applications. The work uses LULESH (a shock hydrodynamics
proxy app) as a testbed, replacing small compute-heavy functions with neural networks and measuring
the resulting speedup and accuracy tradeoffs.

## Start Here

| Document | What it covers |
|----------|---------------|
| [research_overview.md](research_overview.md) | What the research is about, what was tried, and what the key findings were. Read this first. |
| [codebase_guide.md](codebase_guide.md) | How the code is organized: all key files, the config system, dataset format, model layer types, checkpoint system, and utility scripts. |
| [workflow.md](workflow.md) | Step-by-step guide for training a surrogate model: profiling → data collection → hyperparameter search → pruning → evaluation in LULESH. |

## Deep Dives

| Document | What it covers |
|----------|---------------|
| [retrain_study.md](retrain_study.md) | The `retrain-study/` experiment: comparing fine-tuning a trained model vs. training from scratch when moving to a larger domain size. |
| [prune_model_plan.md](prune_model_plan.md) | Design of `prune_model.py`: how dead neurons are detected and how weight matrices are sliced for each layer type. |
| [visualize_activations_design.md](visualize_activations_design.md) | Design of `visualize_activations.py`: forward hook strategy, violin plot layout, outlier filtering approach. |
| [test_evd_model_design.md](test_evd_model_design.md) | Design of `test_evd_model.py`: Jacobian heatmaps, input sweep plots, and model loading strategy for the EVD function. |

## Quick Reference

**Setup**
```bash
uv sync && source .venv/bin/activate
```

**Train a model**
```bash
python main.py --config configs/my_config.yaml
```

**Resume / retrain on new data**
```bash
python main.py --checkpoint logs/<study>/<run>/ --train.epoch 100 --dataset.db_file new.h5
```

**Run hyperparameter search**
```bash
python main.py --tune --config configs/my_config.yaml
```

**Visualize activations → prune → export**
```bash
python visualize_activations.py --checkpoint logs/<study>/<run>/
python prune_model.py --checkpoint logs/<study>/<run>/
python main.py --create-model --checkpoint logs/<study>/<run>/
```
