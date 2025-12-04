import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_sorted_order(data, metric_name):
    subset = data[data['metric'] == metric_name]
    means = subset.groupby(['config', 'round'])['value'].mean().reset_index()
    ranking = means.groupby('config')['value'].max().sort_values(ascending=False)
    return ranking.index.tolist()

def create_plot(metric, errorbar_type, filename, title_suffix, order, all_data, script_dir):
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=all_data[all_data['metric'] == metric],
        x="round", y="value",
        hue="config",
        hue_order=order,
        marker="o", 
        errorbar=errorbar_type,
        palette="tab10"
    )
    title = f"{metric.upper()} per Round - {title_suffix}"
    plt.title(title, fontsize=16)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {filename}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_path = os.path.join(script_dir, "client_*.csv")
    
    print(f"Search for files in: {search_path}")
    files = glob.glob(search_path)
    
    if not files:
        print("No client CSV files found.!")
        return

    data_frames = []
    
    # Regex für Parameter: eta, le (local epochs), total (trees)
    pattern = r"eta_([\d\.]+)_le_(\d+)_total_(\d+)"

    print(f"{len(files)} files found.")

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
                print(f"Fehler bei {file}: {e}")

    if not data_frames:
        print("No data extracted.")
        return

    all_data = pd.concat(data_frames, ignore_index=True)

    all_data['config'] = (
        "η=" + all_data['eta'].astype(str) + 
        ", local-epochs=" + all_data['le'].astype(str) + 
        ", n-trees=" + all_data['total'].astype(str)
    )

    sns.set_theme(style="whitegrid")
   
    auc_order = get_sorted_order(all_data, 'auc')
    f1_order = get_sorted_order(all_data, 'f1')

    create_plot('auc', None, "client_auc_bagging_iid.png", "Average (Bagging - IID)", auc_order, all_data, script_dir)

    create_plot('f1', None, "client_f1_bagging_iid.png", "Average (Bagging - IID)", f1_order, all_data, script_dir)

    print("All plots have been created.")

if __name__ == "__main__":
    main()