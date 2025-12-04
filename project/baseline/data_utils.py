from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import numpy as np
from typing import Tuple

# global Cache
_CACHE = {
    "processed": None,
    "partitions": {}
}

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

def get_processed_data_cached() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Caches the result of data loading and preprocessing.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The cached data tuple:
        (X_train, X_test, y_train, y_test).
    """
    if _CACHE["processed"] is not None:
        # Data is already in cache, return immediately
        return _CACHE["processed"]

    # Execute the original data loading and processing function
    X_train, X_test, y_train, y_test = get_processed_data()

    # Store the result in the cache
    _CACHE["processed"] = (X_train, X_test, y_train, y_test)
    return _CACHE["processed"]

def get_partitioned_data_cached(
    partition_id: int, 
    num_partitions: int, 
    random_state: int = 123,
    partition_test: bool = False 
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves or generates the IID-partitioned data subset for a client, using caching.

    Args:
        partition_id (int): The index of the client/partition (0 to num_partitions - 1).
        num_partitions (int): The total number of clients/partitions.
        random_state (int, optional): Seed used for the deterministic shuffling and 
            splitting of both the train and optional test indices. Defaults to 123.
        partition_test (bool, optional): If True, the global test set is also 
            partitioned and the client receives only a subset (Local Evaluation). 
            If False, the client receives the entire global test set. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The cached/partitioned data:
        (X_train_client, X_test_client, y_train_client, y_test_client).
    """
    
    # Generate unique key for the cache based on all partitioning parameters
    key = (partition_id, num_partitions, random_state, partition_test)
    if key in _CACHE["partitions"]:
        # Cache hit: return the previously computed partition immediately
        return _CACHE["partitions"][key]

    # Cache miss: load the globally processed data
    X_train, X_test, y_train, y_test = get_processed_data_cached()

    rng = np.random.RandomState(random_state)
    
    # Train Partitioning (IID)
    train_indices = np.arange(X_train.shape[0])
    rng.shuffle(train_indices) # Shuffle deterministically
    train_splits = np.array_split(train_indices, num_partitions) # Split into N parts
    
    client_train_idx = train_splits[partition_id]
    X_train_client = X_train[client_train_idx]
    y_train_client = y_train[client_train_idx]

    # Test Partitioning
    if partition_test:
        # Partition the Test Set (Local Evaluation)
        test_indices = np.arange(X_test.shape[0])
        rng.shuffle(test_indices)
        test_splits = np.array_split(test_indices, num_partitions)
        
        client_test_idx = test_splits[partition_id]
        X_test_client = X_test[client_test_idx]
        y_test_client = y_test[client_test_idx]
    else:
        # Default behavior: Global Test Set for all clients (Global Evaluation)
        X_test_client = X_test
        y_test_client = y_test

    # Store result in partition cache and return
    result = (X_train_client, X_test_client, y_train_client, y_test_client)
    _CACHE["partitions"][key] = result
    return result

def get_country_partitioned_data(
    partition_id: int,
    num_partitions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Partitions the training dataset in a Non-IID manner where each client is 
    assigned all training examples belonging to a single 'native-country'.

    Args:
        partition_id (int): The index of the partition/client (0 to num_partitions - 1). 
            This index corresponds to a specific country in the sorted list.
        num_partitions (int): The total number of clients/partitions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]: A tuple containing the data:
        (X_train_client, X_test_global, y_train_client, y_test_global, client_country_name).
    """

    # Load the Adult dataset
    print("--- Load Adult Dataset (country partition) ---")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # Clean the target variable
    y["income"] = y["income"].astype(str).str.replace(".", "", regex=False)
    y_bin = y["income"].apply(lambda x: 0 if "<=50K" in x else 1).values

    # Separate and store the native-country feature (crucial for partitioning)
    countries = X["native-country"].astype(str).values

    # Define numeric and categorical features (same as above)
    numeric_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    print("--- Apply preprocessing (country partition) ---")
    X_processed = preprocessor.fit_transform(X)

    # Train/Test-Split, ensuring the 'countries' array is split along with X and y
    (
        X_train,
        X_test,
        y_train,
        y_test,
        countries_train,
        countries_test,
    ) = train_test_split(
        X_processed,
        y_bin,
        countries,
        test_size=0.2,
        random_state=42,
        stratify=y_bin,
    )

    # Gather all unique countries in the training set
    unique_countries = np.unique(countries_train)
    unique_countries_sorted = np.sort(unique_countries)

    if num_partitions > len(unique_countries_sorted):
        raise ValueError(
            f"num_partitions={num_partitions} exceeds the number of unique countries available ({len(unique_countries_sorted)}) in the dataset."
        )

    # Determine which country belongs to this client based on partition_id
    client_country = unique_countries_sorted[partition_id]

    # Find indices corresponding to this specific country
    client_idx = np.where(countries_train == client_country)[0]

    # Assign client's training data
    X_train_client = X_train[client_idx]
    y_train_client = y_train[client_idx]

    return X_train_client, X_test, y_train_client, y_test, client_country
