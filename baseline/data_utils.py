from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import numpy as np
from typing import Tuple

def get_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the UCI Adult dataset, preprocesses it, and splits it into training and testing sets.

    The preprocessing steps include:
    1. Cleaning the target variable 'income'.
    2. Imputing missing values (median for numeric, most frequent for categorical).
    3. Scaling numeric features using StandardScaler.
    4. One-hot encoding categorical features.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the split data:
        (X_train, X_test, y_train, y_test).
    """
    print("--- Load Adult Dataset ---")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    
    # clean target variable
    y = y.copy()
    y['income'] = y["income"].astype(str).str.replace('.', '', regex=False)
    y_bin = y['income'].apply(lambda x: 0 if '<=50K' in x else 1).values
    
    # define numeric and categorical features
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
    
    print("--- Apply preprocessing ---")
    X_processed = preprocessor.fit_transform(X)
    
    return train_test_split(X_processed, y_bin, test_size=0.2, random_state=42, stratify=y_bin)