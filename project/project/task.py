"""xgboost_quickstart: A Flower / XGBoost app."""

import xgboost as xgb
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from baseline.data_utils import get_partitioned_data_cached


def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients):
    X_train, X_test, y_train, y_test = get_partitioned_data_cached(
        partition_id=partition_id,
        num_partitions=num_clients,
        random_state=123,
    )

    # 2) In DMatrix konvertieren
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

    num_train = len(y_train)
    num_val = len(y_test)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict