import torch
from torch.utils.data import DataLoader, TensorDataset

def load_tabular_data(X_train, y_train, X_test, y_test, batch_size=32):
    # load tabular data into DataLoaders
    trainloader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    return trainloader, testloader

def train_tabular(model, trainloader, epochs, lr, device):
    # trains model on tabular data
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return

def test_tabular(model, testloader, device):
    # eval model
    model.to(device)
    model.eval()
    correct, total, loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss += criterion(outputs, y).item()
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return loss / len(testloader), correct / total
