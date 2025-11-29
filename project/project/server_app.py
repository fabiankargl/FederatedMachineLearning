import torch
import numpy as np
from flwr.serverapp import ServerApp
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp.strategy import FedAvg
from baseline.model_utils import SimpleNN

app = ServerApp()

class FedAvgBalanced(FedAvg):
    """FedAvg mit gleichgewichteter Aggregation (Balanced Weight Averaging)"""
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        weights_list = [res.parameters for res, _ in results]

        averaged_weights = [
            np.mean([w[i] for w in weights_list], axis=0)
            for i in range(len(weights_list[0]))
        ]

        metrics_aggregated = {}
        for res, _ in results:
            for k, v in res.metrics.items():
                metrics_aggregated.setdefault(k, []).append(v)
        metrics_mean = {k: float(np.mean(v)) for k, v in metrics_aggregated.items()}

        return averaged_weights, metrics_mean

@app.main()
def main(grid, context: Context):
    # Input-Dimension als Konfiguration
    input_dim = context.run_config.get("input_dim", 108)  # z.B. 108 für Adult after one-hot

    # Globales Modell initialisieren
    global_model = SimpleNN(input_dim)
    arrays = ArrayRecord(global_model.state_dict())

    # Strategy mit Balanced Aggregation
    strategy = FedAvgBalanced(
        fraction_train=context.run_config.get("fraction_train", 1.0)
    )

    # Trainingskonfiguration aus TOML
    train_config = ConfigRecord({
        "lr": context.run_config.get("lr", 0.01),
        "local_epochs": context.run_config.get("local_epochs", 1),
        "batch_size": context.run_config.get("batch_size", 64)
    })

    # Anzahl Runden aus TOML
    num_rounds = context.run_config.get("num_server_rounds", 50)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )

    torch.save(result.arrays.to_torch_state_dict(), "final_model_balanced41.pt")
    print("Final model saved as final_model_balanced41.pt")
