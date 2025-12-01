import torch
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from baseline.model_utils import SimpleNN
from project.task import load_tabular_data, train_tabular, test_tabular
from baseline.data_utils import get_processed_data

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # load data
    X_train, X_test, y_train, y_test = get_processed_data()
    trainloader, _ = load_tabular_data(X_train, y_train, X_test, y_test)

    # init model
    model = SimpleNN(X_train.shape[1])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train params
    lr = msg.content["config"].get("lr", 0.01)
    epochs = msg.content["config"].get("local_epochs", 1)
    train_tabular(model, trainloader, epochs, lr, device)

    # return updated model
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"num-examples": len(trainloader.dataset)})
    })
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # load data
    X_train, X_test, y_train, y_test = get_processed_data()
    _, testloader = load_tabular_data(X_train, y_train, X_test, y_test)

    # init model
    model = SimpleNN(X_train.shape[1])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # eval
    loss, acc = test_tabular(model, testloader, device)

    # return metrics
    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": loss,
            "eval_acc": acc,
            "num-examples": len(testloader.dataset)
        })
    })
    return Message(content=content, reply_to=msg)
