# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys

notebook_dir = os.getcwd()
project_root_dir = os.path.dirname(notebook_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import h5py
import numpy as np
import math

from dataset import load_dataset
from config import DatasetConfig

# %%
config = DatasetConfig(db_file="../dataset/HGF-all.h5", label="HGF")
dataset = load_dataset(config)

# %%
total_rows = dataset.X.shape[0]
proportion = 0.20 # 20%
sample_count = math.ceil(total_rows * proportion)
train_count = math.ceil(sample_count * 0.8)

random_indices = np.random.choice(total_rows, sample_count, replace=False)

random_indices_train = random_indices[0:train_count]
random_indices_validate = random_indices[train_count:]

# %%
with h5py.File(config.db_file + ".sampled", 'w') as f:
    grp = f.create_group(config.label+"_train")
    grp.create_dataset('input', data=dataset.X[random_indices_train], dtype=np.float32)
    grp.create_dataset('output', data=dataset.Y[random_indices_train], dtype=np.float32)

    grp = f.create_group(config.label+"_validate")
    grp.create_dataset('input', data=dataset.X[random_indices_validate], dtype=np.float32)
    grp.create_dataset('output', data=dataset.Y[random_indices_validate], dtype=np.float32)
