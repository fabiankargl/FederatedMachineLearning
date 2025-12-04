import numpy as np
import csv
import os
from datetime import datetime
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic

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
    fraction_train_cfg = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    strategy_name = context.run_config["strategy"]
    data_distribution = context.run_config["data-distribution"]
    num_supernodes = context.run_config["num-supernodes"]
    local_epochs = context.run_config.get("local-epochs", "unknown")

    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    eta = cfg["params"]["eta"]
    total_trees = num_supernodes * local_epochs * num_rounds
    output_dir = "experiment/global"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{output_dir}/global_{strategy_name}_{num_supernodes}_{data_distribution}_eta_{eta}_le_{local_epochs}_total_{total_trees}.csv"

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize selected XGBoost strategy
    if strategy_name == "bagging":
        strategy = FedXgbBagging(
            fraction_train=fraction_train_cfg,
            fraction_evaluate=fraction_evaluate,
        )
    elif strategy_name == "cyclic":
        # FedXgbCyclic erlaubt nur 0.0 oder 1.0 -> wir erzwingen 1.0
        strategy = FedXgbCyclic(
            fraction_train=1.0,
            fraction_evaluate=1.0,
        )
    else:
        raise ValueError(
            f"Unknown XGBoost strategy '{strategy_name}'. "
            "Use 'bagging' or 'cyclic'."
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
