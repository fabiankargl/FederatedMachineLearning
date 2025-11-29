import torch
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import SimpleNN
from project.task import load_tabular_data, train_tabular, test_tabular
from baseline.data_utils import get_processed_data

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Lade Daten direkt
    X_train, X_test, y_train, y_test = get_processed_data()
    trainloader, _ = load_tabular_data(X_train, y_train, X_test, y_test)

    # Initialisiere Modell
    model = SimpleNN(X_train.shape[1])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Trainingsparameter
    lr = msg.content["config"].get("lr", 0.01)
    epochs = msg.content["config"].get("local_epochs", 1)
    train_tabular(model, trainloader, epochs, lr, device)

    # Rückgabe mit num-examples für FedAvgBalanced
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"num-examples": len(trainloader.dataset)})
    })
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Lade Daten direkt, keine node_config nötig
    X_train, X_test, y_train, y_test = get_processed_data()
    _, testloader = load_tabular_data(X_train, y_train, X_test, y_test)

    # Initialisiere Modell
    model = SimpleNN(X_train.shape[1])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation
    loss, acc = test_tabular(model, testloader, device)

    # Rückgabe mit num-examples
    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": loss,
            "eval_acc": acc,
            "num-examples": len(testloader.dataset)
        })
    })
    return Message(content=content, reply_to=msg)
