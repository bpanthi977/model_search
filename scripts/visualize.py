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
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
import os
import sys
import scipy
import pandas
import math

notebook_dir = os.getcwd()
project_root_dir = os.path.dirname(notebook_dir) # This should resolve to /path/to/project_root/
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import h5py
import numpy as np
from dataset import load_dataset
from config import DatasetConfig

# %%
dataset = load_dataset(DatasetConfig(db_file="../dataset/HGF-20p.h5", label="HGF_validate"))

# %%
# Representative Data

# Position
x0 = dataset.X[:,0]
x1 = dataset.X[:,1]
# Volume Derivative
dvdx0 = dataset.X[:,24]
# Velocity
xd0 = dataset.X[:,48]
yd0 = dataset.X[:,56]
zd0 = dataset.X[:,64]
# Coeff
coeff = dataset.X[:,72]
# Volume
determ = dataset.X[:,73]
# force
fx = dataset.Y[:,0]
fy = dataset.Y[:,8]
fz = dataset.Y[:,23]


# %%
data = [
    {"name": "Position x0", "value":x0},
    {"name": "Position x1", "value":x1},
    {"name": "Volume Derivative dvdx0", "value":dvdx0,  "log":False},
    {"name": "Velocity xd0", "value":xd0, "log":True},
    {"name": "Velocity yd0", "value":yd0, "log":True},
    {"name": "Velocity zd0", "value":zd0,  "bins":20, "log":True},
    {"name": "Coefficient coeff", "value":coeff},
    {"name": "Volume determ", "value":determ},
    {"name": "Force fx", "value":fx, "log":True},
    {"name": "Force fy", "value":fy, "log":True},
    {"name":  "Force fz", "value":fz, "log":True}
]

num_variables = len(data)
num_cols = 3
num_rows = math.ceil(num_variables / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
axes_flat = axes.flatten() # Flatten the 2D array of axes for easy iteration

# --- 5. Iterate Through Variables and Plot on Subplots ---
def plot(i, var_name, values, bins="auto", log=False):
    ax = axes_flat[i]

    plot_instance1 = (
        so.Plot(x=np.abs(values) if log else values)
        .on(ax) # Attach this plot to ax1
        .add(so.Bars(), so.Hist(bins=bins, stat='percent'))
    )

    ax2 = ax.twinx()
    plot_instance2 = (
        so.Plot(x=np.abs(values) if log else values)
        .on(ax2) # Attach this plot to ax1
        .add(so.Line(color='green'), so.Hist(bins=bins, stat='percent', cumulative=True))
        .limit(y=(0, None))
    )

    if log:
        plot_instance1 = plot_instance1.scale(x='log')
        plot_instance2 = plot_instance2.scale(x='log')

    plot_instance1.plot()
    plot_instance2.plot()

    ax2.yaxis.tick_right()
    ax.set_xlabel(var_name)

for (i, d) in enumerate(data):
    plot(i, d["name"], d["value"], log=d.get("log"), bins=d.get("bins") or "auto")
for j in range(num_variables, len(axes_flat)):
    fig.delaxes(axes_flat[j])

fig.tight_layout()
fig.suptitle("Distribution of Dataset Variables", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
plt.savefig("distribution.svg")

# %%
print(f"Variable                  =     Mean (+/-  Std. Dev) [    Minimum,       Maximum]")
print("-" * 81)
for d in data:
    v = d["value"]
    print(f"{d["name"]:25} = {np.mean(v):+.5f} (+/- {np.std(v):9.6f}) [{np.min(v):12.6f}, {np.max(v):12.6f}]")

print(f"Total Datapoints: {data[0]['value'].size}")
