import os
from datetime import datetime
import csv
import xgboost as xgb

def log_round_metrics_to_csv(filename: str, 
                             strategy_name: str, 
                             round_number: int, 
                             metrics: dict) -> None:
    """Logs aggregated metrics from a federated learning round to a CSV file.

    Args:
        filename (str): The path to the output CSV file.
        strategy_name (str): The name of the strategy used in the round.
        round_number (int): The current federated learning round number.
        metrics (dict): A dictionary containing the metrics to be logged.
                        Example: {'auc': 0.85, 'f1': 0.78}.
    """
    file_exists = os.path.isfile(filename)

    headers = ["timestamp", "strategy", "round", "metric", "value"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(headers)

        for key, value in metrics.items():
            writer.writerow([
                timestamp,
                strategy_name,
                round_number,
                key,
                value,
            ])
            
def log_client_metrics_to_csv(filename: str,
                              strategy_name: str,
                              round_number: int,
                              client_id: str,
                              metrics: dict) -> None:
    """Logs metrics from a single client's evaluation to a CSV file.

    Args:
        filename (str): The path to the output CSV file.
        strategy_name (str): The name of the strategy used in the round.
        round_number (int): The current federated learning round number.
        client_id (str): The unique identifier for the client.
        metrics (dict): A dictionary containing the metrics to be logged.
                        Example: {'auc': 0.91, 'f1': 0.88}.
    """
    file_exists = os.path.isfile(filename)

    headers = ["timestamp", "strategy", "round", "client_id", "metric", "value"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(headers)

        for key, value in metrics.items():
            writer.writerow(
                [
                    timestamp,
                    strategy_name,
                    round_number,
                    client_id,
                    key,
                    value,
                ]
            )
            
            
def _local_boost(
    bst_input: xgb.Booster,
    num_local_round: int,
    train_dmatrix: xgb.DMatrix
) -> xgb.Booster:
    """
    Update the XGBoost model based on local training data.
    
    Args:
        bst_input (xgb.Booster): The starting XGBoost model (usually the global 
            model received from the server).
        num_local_round (int): The number of boosting iterations to perform.
        train_dmatrix (xgb.DMatrix): The local training data wrapped in an 
            XGBoost DMatrix.

    Returns:
        xgb.Booster: A new booster object containing only the trees trained 
            during this local session.
    """
    # Update the existing booster for 'num_local_round' iterations
    for _ in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Slicing: Extract only the new trees (from total-N to total)
    bst = bst_input[
        bst_input.num_boosted_rounds() - num_local_round : bst_input.num_boosted_rounds()
    ]
    
    return bst