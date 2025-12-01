import numpy as np
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import train_xgboost
from baseline.data_utils import get_processed_data
from baseline.data_utils import get_partitioned_data
from sklearn.metrics import roc_auc_score, f1_score


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

    # convert model to numpy array
    updated_model_array = np.random.randn(1).astype(np.float32)

    train_preds, train_probs = train_xgboost(X_train, y_train, X_train)
    train_acc = (train_preds == y_train).mean()
    train_auc = roc_auc_score(y_train, train_probs)
    train_f1 = f1_score(y_train, train_preds)



    content = RecordDict({
        # contains model weights
        "arrays": ArrayRecord([updated_model_array]),
        # contains metrics
        "metrics": MetricRecord({"num-examples": len(X_train), "train_acc": train_acc, "train_auc": train_auc, "train_f1": train_f1})
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

    eval_preds, eval_probs = train_xgboost(X_train, y_train, X_test)
    eval_acc = (eval_preds == y_test).mean()
    eval_auc = roc_auc_score(y_test, eval_probs)
    eval_f1 = f1_score(y_test, eval_preds)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_acc": eval_acc,
            "eval_auc": eval_auc,
            "eval_f1": eval_f1,
            "num-examples": len(X_test),
        })
    })
    return Message(content=content, reply_to=msg)
