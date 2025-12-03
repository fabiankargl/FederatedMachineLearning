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

def log_round_metrics_to_csv(strategy_name: str, round_number: int, metrics: dict):
    """Append federated round metrics into a CSV file."""
    
    filename = "test.csv"
    file_exists = os.path.isfile(filename)

    # CSV headers
    headers = ["timestamp", "strategy", "round", "metric", "value"]

    # timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # open and append rows
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        # write header if new file
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
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    
    print("\nWriting metrics to CSV...")
    strategy_name = "FedXgbBagging"

    # Access evaluate metrics from ClientApp side
    # result.evaluate_metrics_clientapp is a dict: {round_number: {metric: value}}
    for rnd, metrics in result.evaluate_metrics_clientapp.items():
        log_round_metrics_to_csv(strategy_name, rnd, metrics)

    print(f"Metrics for {len(result.evaluate_metrics_clientapp)} rounds written to federated_metrics.csv")


    # Save final model to disk
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())

    # Load global model into booster
    bst.load_model(global_model)

    # Save model
    print("\nSaving final model to disk...")
    bst.save_model("final_model.json")