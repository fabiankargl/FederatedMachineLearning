import numpy as np
import csv
import os
from datetime import datetime
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

from project.task import replace_keys

def log_round_metrics_to_csv(filename: str, strategy_name: str, round_number: int, metrics: dict):
    """Append federated round metrics into a CSV file."""
    
    file_exists = os.path.isfile(filename)

    headers = ["timestamp", "strategy", "round", "metric", "value"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(headers)

        # write every metric separately
        for key, value in metrics.items():
            writer.writerow([
                timestamp,
                strategy_name,
                round_number,
                key,
                value,
            ])

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    strategy_name = context.run_config["strategy"]
    data_distribution = context.run_config["data-distribution"]
    num_supernodes = context.run_config["num-supernodes"]
    
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    
    csv_filename = f"federated_metrics_{strategy_name}_{num_supernodes}_{data_distribution}.csv"

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    
    print("\nWriting metrics to CSV...")

    for rnd, metrics in result.evaluate_metrics_clientapp.items():
        log_round_metrics_to_csv(csv_filename, strategy_name, rnd, metrics)

    print(f"Metrics for {len(result.evaluate_metrics_clientapp)} rounds written to {csv_filename}")

    # Save final model to disk
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())

    # Load global model into booster
    bst.load_model(global_model)

    print("\nSaving final model to disk...")
    bst.save_model("final_model.json")