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

def get_partitioned_data(
    partition_id: int,
    num_partitions: int,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gibt nur den Teil der Daten für einen Client zurück.

    partition_id:   welche Partition (0 .. num_partitions-1)
    num_partitions: wie viele Clients insgesamt
    random_state:   sorgt dafür, dass der Split reproduzierbar ist
    """

    # 1) globalen Datensatz laden (einmalige Preprocessing-Logik wiederverwenden)
    X_train, X_test, y_train, y_test = get_processed_data()

    # 2) Indizes des Train-Sets mischen, aber deterministisch
    rng = np.random.RandomState(random_state)
    indices = np.arange(X_train.shape[0])
    rng.shuffle(indices)

    # 3) Indizes in num_partitions ungefährt gleich große Blöcke teilen
    splits = np.array_split(indices, num_partitions)

    # 4) Indizes für diesen Client
    client_idx = splits[partition_id]

    X_train_client = X_train[client_idx]
    y_train_client = y_train[client_idx]

    return X_train_client, X_test, y_train_client, y_test

def get_country_partitioned_data(
    partition_id: int,
    num_partitions: int,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Non-IID: Ein Client = ein Land (native-country).

    partition_id:   0 .. num_partitions-1
    num_partitions: Anzahl Clients (z.B. 41)
    random_state:   für Reproduzierbarkeit
    """

    # 1) Adult-Datensatz laden
    print("--- Load Adult Dataset (country partition) ---")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # 2) Target säubern (gleich wie in get_processed_data)
    y["income"] = y["income"].astype(str).str.replace(".", "", regex=False)
    y_bin = y["income"].apply(lambda x: 0 if "<=50K" in x else 1).values

    # 3) native-country separat merken
    countries = X["native-country"].astype(str).values

    # 4) numeric / categorical features wie oben
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

    # 5) Preprocessing-Pipeline wie in get_processed_data
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

    # 6) Train/Test-Split, countries mit rüberziehen
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

    # 7) alle Länder im Train-Set sammeln
    unique_countries = np.unique(countries_train)
    unique_countries_sorted = np.sort(unique_countries)

    if num_partitions > len(unique_countries_sorted):
        raise ValueError(
            f"num_partitions={num_partitions}, aber nur {len(unique_countries_sorted)} Länder im Datensatz"
        )

    # dieses Land gehört zu diesem Client
    client_country = unique_countries_sorted[partition_id]

    # Indizes für dieses Land
    client_idx = np.where(countries_train == client_country)[0]

    X_train_client = X_train[client_idx]
    y_train_client = y_train[client_idx]

    print(
        f"[Partition] id={partition_id}, country={client_country}, n_train={len(y_train_client)}"
    )

    # Testset bleibt global
    return X_train_client, X_test, y_train_client, y_test, client_country