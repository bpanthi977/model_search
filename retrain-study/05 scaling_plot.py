import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OFFSETS = {
    '#2': 20,
    '#3': 50
}

if __name__ == '__main__':
    if not os.path.exists('summary.csv'):
        exit("summary.csv not found")

    if not os.path.exists('runs.csv'):
        exit("runs.csv not found")

    # Load data
    runs = pd.read_csv('runs.csv')
    summary_df = pd.read_csv('summary.csv')
    
    # Correctly merge summary data with run metadata
    df = summary_df.merge(runs[['run', 'model_type', 'train_type']], on='run')

    os.makedirs('plots', exist_ok=True)

    for model_type in ['#2', '#3']:
        offset = OFFSETS[model_type]
        retrain_epochs = [10, offset] #range(10, offset + 1, 10)

        scratch2 = df[(df['train_type'] == 'S')].copy()
        scratch2['train_type'] = 'S2'
        scratch2['epoch'] = scratch2['epoch'] + offset
        df2 = pd.concat([df, scratch2])
        
        # Horizontal layout: 5 inches width per subplot
        fig_width = 5 * len(retrain_epochs)
        grid_fig, grid_axes = plt.subplots(1, len(retrain_epochs), figsize=(fig_width, 5), constrained_layout=True, sharey=True)

        for (i, retrain_epoch) in enumerate(retrain_epochs):
            ax = grid_axes[i]
            target_epoch = offset + retrain_epoch

            runs_df = df2[
                (df2['model_type'] == model_type) &
                (df2['epoch'] == target_epoch) &
                (df2['train_type'].isin(['S', 'R', 'S2'])) &
                (df2['grid_size'] != 30)
            ].copy()

            if runs_df.empty:
                ax.text(0.5, 0.5, f"No data for epoch {target_epoch}", ha='center')
                ax.set_title(f"Epoch {target_epoch}")
                continue

            # Plot scaling curve with direct hue control and palette
            sns.lineplot(
                data=runs_df, 
                x='grid_size', 
                y='mae', 
                hue='train_type', 
                marker='o', 
                hue_order=['S2', 'S', 'R'],
                palette={'S': 'C1', 'R': 'C2', 'S2': 'C3'},
                ax=ax
            )

            # ax.set_title(f"Scratch: {target_epoch} epochs\nRetrained: {offset}+{retrain_epoch} epochs")
            ax.set_ylabel("Energy MAE")
            ax.tick_params(labelleft=True)
            ax.set_xlabel("Training & Evaluation Grid Size")
            
            # Show ticks only for values with data points
            tick_values = sorted(runs_df['grid_size'].unique())
            ax.set_xticks(tick_values)
            ax.set_xticklabels(tick_values)
            
            # Clean up the legend labels for the final plot
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [f'{retrain_epoch} (Scratch)', f'{target_epoch} (Scratch)', f'{retrain_epoch} (Retrained)'], title='Training Epochs')

        plt.suptitle(f'Scaling Evaluation for EVD Model {model_type}')
        
        # Save plot with a safe filename
        safe_name = model_type.replace('#', 'model_')
        grid_fig.savefig(f"plots/scaling_{safe_name}.png")
        grid_fig.savefig(f"plots/scaling_{safe_name}.pdf")
        grid_fig.savefig(f"plots/scaling_{safe_name}.svg")
        print(f"Plot saved in plots/scaling_{safe_name}.png")
