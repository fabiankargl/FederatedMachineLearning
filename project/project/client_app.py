import warnings
import numpy as np
from sklearn.metrics import f1_score
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from project.task import load_data, replace_keys

warnings.filterwarnings("ignore", category=UserWarning)

# Flower ClientApp
app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix):
    """
    Update trees based on local training data.
    Extract the last N=num_local_round trees for server aggregation (bagging).
    """
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    bst = bst_input[
        bst_input.num_boosted_rounds() - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    # Node config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Run config
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    num_local_round = cfg["local_epochs"]
    data_distribution = cfg.get("data_distribution", "non-iid")

    # Load partitioned data according to IID or Non-IID
    train_dmatrix, _, num_train, _ = load_data(
        partition_id, num_partitions, data_distribution
    )

    global_round = msg.content["config"]["server-round"]

    # First round: train from scratch
    if global_round == 1:
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Load global model
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)

        # Local training
        bst = _local_boost(bst, num_local_round, train_dmatrix)

    # Save model to bytes
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    # Prepare reply message
    model_record = ArrayRecord([model_np])
    metrics = {"num-examples": num_train}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    # Node config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Run config
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    data_distribution = cfg.get("data_distribution", "non-iid")

    # Load partitioned data according to IID or Non-IID
    _, valid_dmatrix, _, num_val = load_data(
        partition_id, num_partitions, data_distribution
    )

    # Load global model
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    # --- AUC über xgboost
    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    auc = float(eval_results.split("\t")[1].split(":")[1])

    # --- F1-Score
    y_prob = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    optimal_threshold = 0.4
    y_pred = (y_prob >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)

    # Prepare reply message
    metrics = {
        "auc": auc,
        "f1": f1,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
