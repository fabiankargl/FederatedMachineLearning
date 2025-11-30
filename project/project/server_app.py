import numpy as np
from flwr.serverapp import ServerApp
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp.strategy import FedAvg

app = ServerApp()


class FedAvgBalancedXGB(FedAvg):

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        arrays_list = [res.parameters for res, _ in results]

        # Equal-weighted averaging
        averaged_weights = [
            np.mean([w[i] for w in arrays_list], axis=0)
            for i in range(len(arrays_list[0]))
        ]

        # Aggregate metrics
        metrics_aggregated = {}
        for res, _ in results:
            if res.metrics:
                for k, v in res.metrics.items():
                    metrics_aggregated.setdefault(k, []).append(v)
        metrics_mean = {k: float(np.mean(v)) for k, v in metrics_aggregated.items()}

        averaged_array_record = ArrayRecord(averaged_weights)
        return averaged_array_record, metrics_mean


@app.main()
def main(grid, context: Context):
    # load config from toml
    num_rounds = context.run_config.get("num-server-rounds") or 1
    fraction_train = context.run_config.get("fraction-train") or 1.0

    initial_array = np.zeros((1,), dtype=np.float32)
    arrays = ArrayRecord([initial_array])

    strategy = FedAvgBalancedXGB()
    train_config = ConfigRecord({
        "fraction_train": fraction_train
    })

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds
    )

    if result.arrays and len(result.arrays) > 0:
        arrays_list = list(result.arrays.values())
        np.save("final_model_balanced_xgb5.npy", arrays_list[0])
