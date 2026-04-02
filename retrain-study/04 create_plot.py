import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RUNS = {
    "30/20260305-195222-ioQW": {'model_id': '#2', 'type': 'O', 'grid': 30},
    "30/20260306-020855-UmhQ": {'model_id': '#3', 'type': 'O', 'grid': 30},

    "40/20260312-193859-IbpD": {'model_id': '#2', 'type': 'S', 'grid': 40},
    "40/20260312-183104-mieu": {'model_id': '#2', 'type': 'R', 'grid': 40},
    "40/20260312-174017-FUQV": {'model_id': '#3', 'type': 'S', 'grid': 40},
    "40/20260312-185952-SXIL": {'model_id': '#3', 'type': 'R', 'grid': 40},

    "50/20260305-192304-tRiJ": {'model_id': '#2', 'type': 'S', 'grid': 50},
    "50/20260305-185353-cGPK": {'model_id': '#2', 'type': 'R', 'grid': 50},
    "50/20260309-151938-rfbA": {'model_id': '#3', 'type': 'S', 'grid': 50},
    "50/20260305-204235-aAiu": {'model_id': '#3', 'type': 'R', 'grid': 50},

    "60/20260312-175059-ThjA": {'model_id': '#2', 'type': 'S', 'grid': 60},
    "60/20260312-182409-QyaY": {'model_id': '#2', 'type': 'R', 'grid': 60},
    "60/20260312-160922-FmUU": {'model_id': '#3', 'type': 'S', 'grid': 60},
    "60/20260312-170036-ZVXo": {'model_id': '#3', 'type': 'R', 'grid': 60},
}

OFFSETS = {
    '#2': 20,
    '#3': 50
}

LABEL_MAP = {
    'O': 'Original 30x',
    'S': 'Scratch',
    'R': 'Retrained'
}
HUE_ORDER = ['Original 30x', 'Scratch', 'Retrained']
GRID_SIZES = [40, 50, 60]

def get_metadata(run_folder):
    """Extracts model ID and type from the folder name."""
    if run_folder in RUNS:
        meta = RUNS[run_folder]
        return meta['model_id'], meta['type']
    exit(f"No metadata for folder {run_folder}")

def get_val_df(run, offset):
    path = os.path.join(run, 'val_loss.csv')
    if not os.path.exists(path):
        exit(f"Warning: {path} not found")

    # Load with integer column names
    df = pd.read_csv(path, header=None).sort_values(0)
    df = df.rename(columns={0: 'epoch', 1: 'val_loss'})

    # Apply offset for validation plots
    df = df[df['epoch'] > offset]
    df['run'] = run
    return df

def get_runs(grid_size, model_type):
    runs = []
    for (r, info) in RUNS.items():
        if info['grid'] == grid_size and info['model_id'] == model_type:
            runs.append(r)

    return runs

def draw_connector(ax, data, x_col, y_col):
    """Draw dotted line from last point of blue (Original 30x) to first point of green (Retrained)."""
    import matplotlib.colors as mcolors
    blue = mcolors.to_rgb(sns.color_palette()[0])  # C0
    green = mcolors.to_rgb(sns.color_palette()[2])  # C2
    mid_color = tuple((b + g) / 2 for b, g in zip(blue, green))

    orig = data[data['type'] == 'Original 30x'].sort_values(x_col)
    ret = data[data['type'] == 'Retrained'].sort_values(x_col)

    if orig.empty or ret.empty:
        return

    orig_last = orig.groupby(x_col)[y_col].mean().reset_index().iloc[-1]
    ret_first = ret.groupby(x_col)[y_col].mean().reset_index().iloc[0]

    ax.plot([orig_last[x_col], ret_first[x_col]],
            [orig_last[y_col], ret_first[y_col]],
            linestyle=':', color=mid_color, linewidth=1.5)

def update_legend(ax, grid_size):
    legend = ax.get_legend()
    if legend is None:
        return
    rename = {
        'Original 30x': 'Grid 30',
        'Scratch': f'Grid {grid_size}',
        'Retrained': f'Grid 30 - {grid_size}',
    }
    for text in legend.get_texts():
        text.set_text(rename.get(text.get_text(), text.get_text()))
    legend.set_title('Training Dataset')

def ensure_list(l):
    if isinstance(l, list):
        return l
    else:
        return [l]

if __name__ == '__main__':
    if not os.path.exists('summary.csv'):
        exit("summary.csv not found")

    df = pd.read_csv('summary.csv')
    # Apply metadata to summary df
    meta_cols = df['run'].apply(get_metadata)
    df['model_id'], df['type'] = zip(*meta_cols)

    # Load and merge validation data
    val_list = []
    for run, info in RUNS.items():
        if info['type'] == 'R':
            offset = OFFSETS[info['model_id']]
        else:
            offset = 0

        vdf = get_val_df(run, offset)
        vdf['run'] = run
        vdf['model_id'] = info['model_id']
        vdf['type'] = info['type']
        vdf['grid'] = info['grid']
        val_list.append(vdf)

    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()

    # Apply labels and categorical type for consistent ordering and coloring
    sns.set_palette(['C0', 'C1', 'C2'])
    for d in [df, val_df]:
        if not d.empty:
            d['type'] = d['type'].map(LABEL_MAP).astype(pd.CategoricalDtype(categories=HUE_ORDER, ordered=True))

    os.makedirs('plots', exist_ok=True)

    for grid_size in GRID_SIZES:
        models = ['#3']#list(OFFSETS.keys())
        grid_fig, grid_axes = plt.subplots(len(models), 3, figsize=(18, 5 * len(models)), constrained_layout=True)
        grid_axes = ensure_list(grid_axes)

        for i, mid in enumerate(models):
            relevant_runs = get_runs(30, mid) + get_runs(grid_size, mid)
            model_df = df[df['run'].isin(relevant_runs)]
            #model_val_df = val_df[(val_df['model_id'] == mid) & ((val_df['grid'] == 30) | (val_df['grid'] == grid_size))]
            model_val_df = val_df[val_df['run'].isin(relevant_runs)]

            # Plot Energy MAE 30
            ax = grid_axes[i][0]
            sns.lineplot(data=model_df[model_df['grid_size'] == 30], x='epoch', y='mae', hue='type', ax=ax, marker='o')
            draw_connector(ax, model_df[model_df['grid_size'] == 30], 'epoch', 'mae')
            update_legend(ax, grid_size)
            ax.set_title(f"Energy MAE evaluated on Grid 30")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Energy MAE")

            # Plot Energy MAE 'grid'
            ax = grid_axes[i][1]
            sns.lineplot(data=model_df[model_df['grid_size'] == grid_size], x='epoch', y='mae', hue='type', ax=ax, marker='o')
            draw_connector(ax, model_df[model_df['grid_size'] == grid_size], 'epoch', 'mae')
            update_legend(ax, grid_size)
            ax.set_title(f"Energy MAE evaluated on Grid {grid_size}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Energy MAE")

            # Plot Validation Loss curve
            ax = grid_axes[i][2]
            sns.lineplot(data=model_val_df, x='epoch', y='val_loss', hue='type', ax=ax)
            draw_connector(ax, model_val_df, 'epoch', 'val_loss')
            update_legend(ax, grid_size)
            ax.set_title(f"Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_yscale('log')

        #plt.suptitle("Training Summary Grid", fontsize=16)
        grid_fig.savefig(f"plots/summary_grid_30-{grid_size}.png")
        grid_fig.savefig(f"plots/summary_grid_30-{grid_size}.pdf")
        grid_fig.savefig(f"plots/summary_grid_30-{grid_size}.svg")
        print(f"Plots saved in plots/summary_grid_30-{grid_size}.png")
