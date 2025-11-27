import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

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
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='income', kde=True, bins=30, palette='coolwarm')
plt.title('Age distribution by income')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='income', y='hours-per-week', data=df, hue='income', palette='Set2', legend=False)
plt.title('Working hours per week vs. income')
plt.show()

def plot_categorical_relation(column, df_data, rotation=45):
    plt.figure(figsize=(12, 6))
    order = df_data[column].value_counts().index
    sns.countplot(x=column, hue='income', data=df_data, order=order, palette='Paired')
    plt.title(f'Influence from "{column}" on the income')
    plt.xticks(rotation=rotation)
    plt.legend(title='Income', loc='upper right')
    plt.show()

plot_categorical_relation('education', df)

plot_categorical_relation('occupation', df, rotation=90)

plot_categorical_relation('relationship', df)

numeric_df = df.select_dtypes(include=['int64', 'float64'])

corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation matrix of the numerical features')
plt.show()

print("Key insights for your model:")
print("1. Imbalanced Data")
print("2. Missing values ​​('?') occur primarily in 'workclass', 'occupation' and 'native-country'")