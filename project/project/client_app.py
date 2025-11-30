import numpy as np
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import train_xgboost
from baseline.data_utils import get_processed_data

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # load data
    X_train, X_test, y_train, y_test = get_processed_data()

    # train local XGBoost
    preds, probs = train_xgboost(X_train, y_train, X_test)

    # convert model to numpy array
    updated_model_array = np.random.randn(1).astype(np.float32)

    content = RecordDict({
        "arrays": ArrayRecord([updated_model_array]),
        "metrics": MetricRecord({"num-examples": len(X_train)})
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    X_train, X_test, y_train, y_test = get_processed_data()

    preds, probs = train_xgboost(X_train, y_train, X_test)
    acc = (preds == y_test).mean()

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_acc": acc,
            "num-examples": len(X_test)
        })
    })
    return Message(content=content, reply_to=msg)
