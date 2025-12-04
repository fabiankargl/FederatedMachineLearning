import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import glob
import os

def plot_and_save(
    strategy_name: str, 
    filename: str, 
    df_all: pd.DataFrame, 
    script_dir: str
) -> None:
    """
    Filters the dataset for a specific strategy and generates side-by-side plots 
    for AUC and F1 scores.
    
    Args:
        strategy_name (str): The name of the strategy to filter by.
        filename (str): The filename for the saved plot image.
        df_all (pd.DataFrame): The DataFrame containing aggregated data from all CSV files.
        script_dir (str): The absolute path to the directory where the plot will be saved.
    """
    df_strat = df_all[df_all['strategy'] == strategy_name]
        
    if df_strat.empty:
        print(f"No data found for strategy: {strategy_name}")
        return

    _, axes = plt.subplots(1, 2, figsize=(18, 7))
        
    sns.lineplot(
        data=df_strat[df_strat['metric'] == 'auc'],
        x='round', y='value', hue='legend_label', style='legend_label',
        markers=True, dashes=False, ax=axes[0], linewidth=2
    )
    axes[0].set_title(f'{strategy_name.capitalize()} (Global): AUC', fontsize=14)
    axes[0].set_ylabel('AUC')
    axes[0].set_xlabel('Round')
    axes[0].legend(title="Configuration")
        
    sns.lineplot(
        data=df_strat[df_strat['metric'] == 'f1'],
        x='round', y='value', hue='legend_label', style='legend_label',
        markers=True, dashes=False, ax=axes[1], linewidth=2
    )
    axes[1].set_title(f'{strategy_name.capitalize()} (Global): F1 Score', fontsize=14)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_xlabel('Round')
    axes[1].legend(title="Configuration")

    plt.suptitle(f'{strategy_name.capitalize()} Strategy Global Performance', fontsize=16)
    plt.tight_layout()
        
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved: {save_path}")

def create_strategy_plots() -> None:
    """
    Locates global strategy CSV files, aggregates their data, and triggers plotting.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, "*.csv")

    print(f"Searching for files in: {search_path}")
    
    files = glob.glob(search_path)
    
    data = []
    pattern = r"global_([a-zA-Z]+)_(\d+)_([a-zA-Z-]+)_eta"

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
                df['legend_label'] = f"{clients} Clients ({dist.upper()})"
                data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not data:
        print("No matching data found. Please check if the CSV files are in the script directory.")
        return

    df_all = pd.concat(data, ignore_index=True)
    
    df_all = df_all.sort_values(by=['clients', 'distribution'])

    sns.set_theme(style="whitegrid")

    plot_and_save('bagging', 'bagging_performance.png', df_all, script_dir)
    plot_and_save('cyclic', 'cyclic_performance.png', df_all, script_dir)

if __name__ == "__main__":
    create_strategy_plots()