import numpy as np
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import train_xgboost
from baseline.data_utils import get_processed_data
from baseline.data_utils import get_partitioned_data

app = ClientApp()

# train function
@app.train()
def train(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # load data
    X_train, X_test, y_train, y_test = get_partitioned_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
    )

    # load config
    config = context.run_config
    fraction_train = config.get("fraction_train", 1.0)

    # train local xgboost
    preds, probs = train_xgboost(X_train, y_train, X_test)

    # convert model to numpy array
    updated_model_array = np.random.randn(1).astype(np.float32)

    train_acc = (preds == y_test).mean()

    content = RecordDict({
        # contains model weights
        "arrays": ArrayRecord([updated_model_array]),
        # contains metrics
        "metrics": MetricRecord({"num-examples": len(X_train), "train_acc": train_acc})
    })
    return Message(content=content, reply_to=msg)


# eval function
@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    X_train, X_test, y_train, y_test = get_partitioned_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
    )

    preds, probs = train_xgboost(X_train, y_train, X_test)
    acc = (preds == y_test).mean()

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_acc": acc,
            "num-examples": len(X_test),
        })
    })
    return Message(content=content, reply_to=msg)
