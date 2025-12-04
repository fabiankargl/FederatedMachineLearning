import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_sorted_order(data, metric_name):
    subset = data[data['metric'] == metric_name]
    ranking = subset.groupby('config')['value'].max().sort_values(ascending=False)
    return ranking.index.tolist()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_path = os.path.join(script_dir, "*.csv")
    
    print(f"Search for files: {search_path}")
    files = glob.glob(search_path)
    
    if not files:
        print("No CSV-files found!")
        return

    data_frames = []
    
    pattern = r"eta_([\d\.]+)_le_(\d+)_total_(\d+)"

    print(f"{len(files)} files found!")

    for file in files:
        match = re.search(pattern, file)
        if match:
            eta = float(match.group(1))
            le = int(match.group(2))
            total = int(match.group(3))
            
            try:
                df = pd.read_csv(file)
                df['eta'] = eta
                df['le'] = le
                df['total'] = total
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        else:
            pass

    if not data_frames:
        print("No valid data could be extracted.")
        return

    all_data = pd.concat(data_frames, ignore_index=True)

    all_data['config'] = (
        "η=" + all_data['eta'].astype(str) + 
        ", local-epochs=" + all_data['le'].astype(str) + 
        ", n-trees=" + all_data['total'].astype(str)
    )
    sns.set_theme(style="whitegrid")
    
    auc_order = get_sorted_order(all_data, 'auc')

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=all_data[all_data['metric'] == 'auc'],
        x="round", y="value",
        hue="config",
        hue_order=auc_order, 
        marker="o",
        palette="tab10"
    )
    plt.title('AUC per Round - Cyclic', fontsize=16)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("AUC", fontsize=12)
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "auc_plot_global_cyclic.png"), dpi=300)
    
    f1_order = get_sorted_order(all_data, 'f1')

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=all_data[all_data['metric'] == 'f1'],
        x="round", y="value",
        hue="config",
        hue_order=f1_order,
        marker="o",
        palette="tab10"
    )
    plt.title('F1 Score per Round - Cyclic', fontsize=16)
    plt.xlabel("Runde", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "f1_plot_global_cyclic.png"), dpi=300)

    print(f"Done! Sorted plots have been placed in the folder '{script_dir}'.")

if __name__ == "__main__":
    main()