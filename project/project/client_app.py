import csv
import os
import numpy as np
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import train_xgboost
from baseline.data_utils import get_partitioned_data, get_country_partitioned_data
from sklearn.metrics import roc_auc_score, f1_score

app = ClientApp()

# Anzahl der lokalen Modelle für Bagging
NUM_BAGGING_MODELS = 3

@app.train()
def train(msg: Message, context: Context):
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

    print(f"[TRAIN] client={partition_id}, mode={mode}, country={client_country}, n={len(y_train)}")

    # Bagging: mehrere XGBoost-Modelle trainieren
    bagging_preds = []
    bagging_probs = []
    model_arrays = []

    for i in range(NUM_BAGGING_MODELS):
        preds, probs = train_xgboost(X_train, y_train, X_train)
        bagging_preds.append(preds)
        bagging_probs.append(probs)

        # Dummy Array für FedAvg
        model_arrays.append(np.random.randn(1).astype(np.float32))

    # Ensemble-Voting / Durchschnitt
    ensemble_preds = np.mean(np.stack(bagging_preds), axis=0).round()
    ensemble_probs = np.mean(np.stack(bagging_probs), axis=0)

    train_acc = float((ensemble_preds == y_train).mean())
    train_f1  = float(f1_score(y_train, ensemble_preds))
    train_auc = float(roc_auc_score(y_train, ensemble_probs))

    # Arrays für FedAvg
    updated_model_array = np.mean(np.stack(model_arrays), axis=0)

    content = RecordDict({
        "arrays": ArrayRecord([updated_model_array]),
        "metrics": MetricRecord({
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_auc": train_auc,
            "num-examples": int(len(y_train)),
        }),
    })

    return Message(content=content, reply_to=msg)


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

    # Lokales Bagging
    bagging_preds = []
    bagging_probs = []

    for i in range(NUM_BAGGING_MODELS):
        preds, probs = train_xgboost(X_train, y_train, X_test)
        bagging_preds.append(preds)
        bagging_probs.append(probs)

    ensemble_preds = np.mean(np.stack(bagging_preds), axis=0).round()
    ensemble_probs = np.mean(np.stack(bagging_probs), axis=0)

    eval_acc = (ensemble_preds == y_test).mean()
    eval_auc = roc_auc_score(y_test, ensemble_probs)
    eval_f1 = f1_score(y_test, ensemble_preds)

    # CSV Logging
    csv_path = "client_metrics.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "client_id", "mode", "country", "n_train", "n_test", "acc", "f1", "auc"
            ])
        writer.writerow([
            partition_id, mode, client_country,
            len(y_train), len(y_test),
            float(eval_acc), float(eval_f1), float(eval_auc)
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
