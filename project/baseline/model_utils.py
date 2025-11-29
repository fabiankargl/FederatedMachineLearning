import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

# XGBoost
def train_xgboost(X_train: np.ndarray, 
                  y_train: np.ndarray, 
                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains an XGBoost classifier on the given data.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data for prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - preds (np.ndarray): The predicted labels for the test set.
            - probs (np.ndarray): The prediction probabilities for the positive class.
    """
    print("\n--- Start training XGBoost ---")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=500,        
        learning_rate=0.05,      
        max_depth=6,             
        subsample=0.9,           
        colsample_bytree=0.7,    
        scale_pos_weight=3       
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("--- Finished training XGBoost ---")
    return preds, probs

# Random Forest
def train_random_forest(X_train: np.ndarray, 
                        y_train: np.ndarray, 
                        X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains a Random Forest classifier on the given data.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data for prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - preds (np.ndarray): The predicted labels for the test set.
            - probs (np.ndarray): The prediction probabilities for the positive class.
    """
    print("\n--- Start training Random Forest ---")
    model = RandomForestClassifier(
        n_estimators=200,        
        min_samples_split=2,     
        min_samples_leaf=2,      
        max_depth=None,          
        class_weight='balanced', 
        bootstrap=True,          
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("--- Finished training Random Forest ---")
    return preds, probs 

# Neural Network
class SimpleNN(nn.Module):
    """
    A simple feed-forward neural network for binary classification.

    The network consists of two hidden layers with ReLU activation and Dropout,
    and an output layer with two neurons for binary classification.
    """
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def train_neural_network(X_train: np.ndarray, 
                         y_train: np.ndarray, 
                         X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains a simple neural network using PyTorch.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data for prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the predicted labels and probabilities.
    """
    print("\n--- Start training Neural Network ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Network training on device: {device}")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SimpleNN(X_train.shape[1]).to(device=device)
    
    class_counts = np.bincount(y_train)
    weights = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights) 
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device))
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)
    
    print("--- Finished training Neural Network ---")
    return preds.cpu().numpy(), probs.cpu().numpy()