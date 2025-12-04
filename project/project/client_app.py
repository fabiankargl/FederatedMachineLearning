import warnings
import numpy as np
import os
from sklearn.metrics import f1_score
import xgboost as xgb
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from project.task import load_data, replace_keys
from project.utils import log_client_metrics_to_csv, _local_boost

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    # Node-specific configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_distribution = context.run_config["data-distribution"]
    
    # Run configuration from the server
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    num_local_round = cfg["local_epochs"]

    # Load partitioned data according to IID or Non-IID
    train_dmatrix, _, num_train, _ = load_data(
        partition_id, num_partitions, data_distribution
    )

    global_round = msg.content["config"]["server-round"]

    # First round: train a new model from scratch
    if global_round == 1:
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Subsequent rounds: load global model and perform local boosting
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)

        # Local training
        bst = _local_boost(bst, num_local_round, train_dmatrix)

    # Save model to a byte array
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    # Prepare reply message with the new model and number of examples
    model_record = ArrayRecord([model_np])
    metrics = {"num-examples": num_train}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    # Node-specific configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_distribution = context.run_config["data-distribution"]

    cfg = replace_keys(unflatten_dict(context.run_config))
    # XGBoost parameters
    params = cfg["params"]
    eta = cfg["params"]["eta"]

    strategy_name = context.run_config.get("strategy", "unknown")
    num_supernodes = context.run_config.get("num-supernodes", "unknown")
    server_round = msg.content["config"]["server-round"]
    local_epochs = context.run_config.get("local-epochs", "unknown")
    num_server_rounds = context.run_config.get("num-server-rounds", 10)
    total_trees = num_supernodes * local_epochs * num_server_rounds
    
    # Load local validation data
    _, valid_dmatrix, _, num_val = load_data(
        partition_id, num_partitions, data_distribution
    )

    # Load global model from server
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    # AUC
    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    auc = float(eval_results.split("\t")[1].split(":")[1])

    # F1 Score
    y_prob = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    optimal_threshold = 0.4
    y_pred = (y_prob >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "auc": auc,
        "f1": f1,
        "num-examples": num_val,
    }

    # Log client-specific metrics to a local CSV file
    output_dir = f"experiment/client/{data_distribution}/{strategy_name}"
    os.makedirs(output_dir, exist_ok=True)
    client_csv_filename = (
    f"{output_dir}/client_{strategy_name}_{num_supernodes}_{data_distribution}_eta_{eta}_le_{local_epochs}_total_{total_trees}.csv")
    client_id = f"client_{partition_id}"

    log_client_metrics_to_csv(
        filename=client_csv_filename,
        strategy_name=strategy_name,
        round_number=server_round,
        client_id=client_id,
        metrics=metrics,
    )

    # Prepare and send response
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
