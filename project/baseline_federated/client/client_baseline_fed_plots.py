import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import glob
import os
from typing import List

def plot_and_save(
    strategy_name: str, 
    filename: str, 
    df_all: pd.DataFrame, 
    script_dir: str
) -> None:
    """
    Generates side-by-side line plots for AUC and F1 Score showing the average 
    performance across clients for a specific strategy.

    Args:
        strategy_name (str): The name of the strategy to plot (e.g., 'bagging', 'cyclic').
        filename (str): The name for the output image file (e.g., 'bagging_performance.png').
        df_all (pd.DataFrame): The combined DataFrame containing data from all loaded client CSVs.
        script_dir (str): The absolute path to the directory where plots should be saved.
    """
    df_strat = df_all[df_all['strategy'] == strategy_name]
        
    if df_strat.empty:
        print(f"No data found for strategy: {strategy_name}")
        return

    _, axes = plt.subplots(1, 2, figsize=(18, 7))
        
    sns.lineplot(
        data=df_strat[df_strat['metric'] == 'auc'],
        x='round', y='value', hue='legend_label', style='legend_label',
        markers=True, dashes=False, ax=axes[0], linewidth=2,
        errorbar='ci'
    )
    axes[0].set_title(f'{strategy_name.capitalize()} (Clients): AUC', fontsize=14)
    axes[0].set_ylabel('AUC')
    axes[0].set_xlabel('Round')
    axes[0].legend(title="Configuration")
        
    sns.lineplot(
        data=df_strat[df_strat['metric'] == 'f1'],
        x='round', y='value', hue='legend_label', style='legend_label',
        markers=True, dashes=False, ax=axes[1], linewidth=2,
        errorbar='ci'
    )
    axes[1].set_title(f'{strategy_name.capitalize()} (Clients): F1 Score', fontsize=14)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_xlabel('Round')
    axes[1].legend(title="Configuration")

    plt.suptitle(f'{strategy_name.capitalize()} Strategy - Client Performance (Average)', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved: {save_path}")

def create_client_plots() -> None:
    """
    Locates client-level CSV files, extracts metadata from filenames, and generates 
    performance comparison plots showing the average client performance.
    """
    # 1. Determine the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, "client_*.csv")

    print(f"Searching for client files in: {search_path}")
    
    files = glob.glob(search_path)
    
    data: List[pd.DataFrame] = []
    pattern = r"client_([a-zA-Z]+)_(\d+)_([a-zA-Z-]+)_eta"

    print(f"{len(files)} files found.")

    for f in files:
        filename = os.path.basename(f)
        match = re.search(pattern, filename)
        
        if match:
            strategy = match.group(1)
            clients = match.group(2)
            dist = match.group(3)
            
            try:
                df = pd.read_csv(f)
                df['strategy'] = strategy
                df['clients'] = int(clients)
                df['distribution'] = dist
                # Create label for the legend
                df['legend_label'] = f"{clients} Clients ({dist.upper()})"
                data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: Filename '{filename}' did not match the expected pattern and was skipped.")

    if not data:
        print("No matching client data found.")
        return

    df_all = pd.concat(data, ignore_index=True)
    
    df_all = df_all.sort_values(by=['clients', 'distribution'])

    sns.set_theme(style="whitegrid")

    plot_and_save('bagging', 'client_bagging_performance.png', df_all, script_dir)
    plot_and_save('cyclic', 'client_cyclic_performance.png', df_all, script_dir)

if __name__ == "__main__":
    create_client_plots()