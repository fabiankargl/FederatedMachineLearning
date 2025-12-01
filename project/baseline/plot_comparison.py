import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'Model_Full': ['XGBoost (n=500)', 'Random Forest (n=200)', 'Neural Network', 
                   'XGBoost (n=100)', 'XGBoost (n=300)', 'XGBoost (n=200)', 
                   'Random Forest (n=300)', 'Random Forest (n=100)'],
    'F1-Score': [0.7188, 0.7036, 0.6690, 0.7106, 0.7095, 0.6908, 0.6951, 0.6947],
    'AUC':      [0.9315, 0.9176, 0.9138, 0.9276, 0.9265, 0.9238, 0.9139, 0.9135]
}

df = pd.DataFrame(data)

def get_family(name: str) -> str:
    """
    Determines the model family from a full model name string.

    Args:
        name (str): The full name of the model.

    Returns:
        str: The family name of the model.
    """
    if 'XGBoost' in name:
        return 'XGBoost'
    if 'Random Forest' in name:
        return 'Random Forest'
    return 'Neural Network'

df['Family'] = df['Model_Full'].apply(get_family)
df['Short_Label'] = df['Model_Full'].str.replace('XGBoost ', '').str.replace('Random Forest ', '')

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))

sns.scatterplot(
    data=df, 
    x="AUC", 
    y="F1-Score", 
    hue="Family",      
    style="Family",    
    s=200,             
    palette="deep",    
    alpha=0.8,         
    edgecolor="black",
    ax=ax
)

texts = []
for i in range(df.shape[0]):
    row = df.iloc[i]

    label_text = f"{row['Short_Label']}"
    
    if row['F1-Score'] == df['F1-Score'].max():
        t = ax.text(row['AUC'], row['F1-Score'] + 0.001, label_text, 
                    fontsize=11, color='black')
    else:
        t = ax.text(row['AUC'], row['F1-Score'] + 0.001, label_text, 
                    fontsize=9, color='#444444')
    texts.append(t)

plt.title('Model-Comparison', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('AUC Score', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Model-Type")

ax.grid(True, linestyle=':', alpha=0.7)

x_min, x_max = df['AUC'].min(), df['AUC'].max()
y_min, y_max = df['F1-Score'].min(), df['F1-Score'].max()
padding_x = (x_max - x_min) * 0.2
padding_y = (y_max - y_min) * 0.2

plt.xlim(x_min - padding_x, x_max + padding_x)
plt.ylim(y_min - padding_y, y_max + padding_y)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()