import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from typing import Tuple, List, Any

def plot_categorical_relation(
    column: str, 
    df_data: pd.DataFrame, 
    rotation: int = 45
) -> None:
    """
    Generates a count plot showing the distribution of the target variable ('income') 
    across categories of a specified feature column.

    Args:
        column (str): The name of the categorical column in df_data to plot on the x-axis.
        df_data (pd.DataFrame): The DataFrame containing the data, including the 'income' target column.
        rotation (int, optional): The rotation angle for the x-axis labels. Defaults to 45.
    """
    plt.figure(figsize=(12, 6))
    # Order categories by their total count
    order = df_data[column].value_counts().index
    sns.countplot(x=column, hue='income', data=df_data, order=order, palette='Paired')
    
    plt.title(f'Income distribution by "{column}"')
    plt.xticks(rotation=rotation)
    plt.legend(title='Income', loc='upper right')
    plt.xlabel(column.replace('-', ' ').capitalize()) 
    plt.tight_layout()
    plt.show()

print("--- Load Data... ---")
adult = fetch_ucirepo(id=2) 
adult_features = adult.data.features
adult_targets = adult.data.targets

# create a combined dataset
df = pd.concat([adult_features, adult_targets], axis=1)

# replace ? values with NaN
df.replace('?', np.nan, inplace=True)

# remove . from target variable income
df['income'] = df['income'].astype(str).str.replace('.', '', regex=False)

print("\n--- Missing Values per Column ---")
print(df.isnull().sum())

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

plt.figure(figsize=(6, 4))
sns.countplot(x='income', data=df, hue='income', palette='viridis', legend=False)
plt.title('Distribution of the target variable (income)')
plt.xlabel('Income bracket')
plt.ylabel('Number')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='income', kde=True, bins=30, palette='coolwarm')
plt.title('Age distribution by income')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='income', y='hours-per-week', data=df, hue='income', palette='Set2', legend=False)
plt.title('Working hours per week vs. income')
plt.tight_layout()
plt.show()

plot_categorical_relation('education', df)

plot_categorical_relation('occupation', df, rotation=90)

plot_categorical_relation('relationship', df)

numeric_df = df.select_dtypes(include=['int64', 'float64'])

corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation matrix of the numerical features')
plt.tight_layout()
plt.show()

print("Key insights for your model:")
print("1. Imbalanced Data")
print("2. Missing values ​​('?') occur primarily in 'workclass', 'occupation' and 'native-country'")