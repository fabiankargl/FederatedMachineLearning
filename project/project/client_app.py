import csv
import os

import numpy as np
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import train_xgboost
from baseline.data_utils import get_processed_data
from baseline.data_utils import get_partitioned_data
from sklearn.metrics import roc_auc_score, f1_score
from baseline.data_utils import (
    get_partitioned_data,            # IID
    get_country_partitioned_data     # non-IID
)


app = ClientApp()

# train function
@app.train()
def train(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # mode aus run_config: "iid" oder "country"
    mode = context.run_config.get("mode", "iid")

    if mode == "country":
        X_train, X_test, y_train, y_test, client_country = get_country_partitioned_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
        )
    else:
        X_train, X_test, y_train, y_test = get_partitioned_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
        )
        client_country = "IID"

    print(f"[TRAIN] client={partition_id}, mode={mode}, country={client_country}, n={len(y_train)}")

    # lokal trainieren (hier nur auf X_train)
    preds_train, probs_train = train_xgboost(X_train, y_train, X_train)

    train_acc = float((preds_train == y_train).mean())
    train_f1  = float(f1_score(y_train, preds_train))
    train_auc = float(roc_auc_score(y_train, probs_train))

    # Dummy-ArrayRecord, damit FedAvg glücklich ist
    #  kannst du hier die echten Gewichte reingeben)
    updated_model_array = np.random.randn(1).astype(np.float32)

    content = RecordDict({
        "arrays": ArrayRecord([updated_model_array]),          # wichtig für FedAvg
        "metrics": MetricRecord({
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_auc": train_auc,
            "num-examples": int(len(y_train)),
        }),
    })

    return Message(content=content, reply_to=msg)


# eval function
@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    mode = context.run_config.get("mode", "iid")

    if mode == "country":
        X_train, X_test, y_train, y_test, client_country = get_country_partitioned_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
        )

    else:
        X_train, X_test, y_train, y_test = get_partitioned_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
        )
        client_country = "IID"

    eval_preds, eval_probs = train_xgboost(X_train, y_train, X_test)
    eval_acc = (eval_preds == y_test).mean()
    eval_auc = roc_auc_score(y_test, eval_probs)
    eval_f1 = f1_score(y_test, eval_preds)

    print(
        f"[EVAL-CLIENt] client={partition_id}, mode={mode}, country={client_country}, n={len(y_train)}, "
        f"n_train={len(y_train)}, n_test={len(y_test)}, "
        f"acc={eval_acc:.4f}, f1={eval_f1:.4f}, auc={eval_auc:.4f}"
    )

    csv_path ="client_metrics.csv"

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "client_id",
                "mode",
                "country",
                "n_train",
                "n_test",
                "acc",
                "f1",
                "auc"])

        writer.writerow([
            partition_id,
            mode,
            client_country,
            len(y_train),
            len(y_test),
            float (eval_acc),
            float (eval_f1),
            float(eval_auc),
        ])

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_acc": eval_acc,
            "eval_auc": eval_auc,
            "eval_f1": eval_f1,
            "num-examples": len(X_test),
        })
    })
    return Message(content=content, reply_to=msg)