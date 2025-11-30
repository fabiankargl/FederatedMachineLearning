import torch
import numpy as np
from flwr.serverapp import ServerApp
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp.strategy import FedAvg
from baseline.model_utils import SimpleNN

app = ServerApp()

class FedAvgBalanced(FedAvg):
    # FedAvg with equal aggregation (Balanced Weight Averaging)
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
    # load config from toml
    lr = context.run_config.get("lr") or 0.01
    local_epochs = context.run_config.get("local-epochs") or 1
    batch_size = context.run_config.get("batch-size") or 64
    fraction_train = context.run_config.get("fraction-train") or 1.0
    num_rounds = context.run_config.get("num-server-rounds") or 1
    input_dim = context.run_config.get("input_dim") or 108

    # init global model
    global_model = SimpleNN(input_dim)
    arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvgBalanced(fraction_train=fraction_train)

    # trainconfig for clients
    train_config = ConfigRecord({
        "lr": lr,
        "local_epochs": local_epochs,
        "batch_size": batch_size
    })

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )

    torch.save(result.arrays.to_torch_state_dict(), "final_model_balanced15.pt")
