import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score 
from ucimlrepo import fetch_ucirepo

def get_processed_data():
    print("--- Load Adult Dataset ---")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    
    # clean target variable
    y = y.copy()
    y['income'] = y["income"].astype(str).str.replace('.', '', regex=False)
    y_bin = y['income'].apply(lambda x: 0 if '<=50K' in x else 1).values
    
    # define features
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
    
    print("--- Apply preprocessing ---")
    X_processed = preprocessor.fit_transform(X)
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train, X_test):
    print("\n--- Train XGBoost ---")
    
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
    return preds, probs

def train_random_forest(X_train, y_train, X_test):
    print("\n--- Training Random Forest ---")
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
    return preds, probs 

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
    
    
def train_neural_network(X_train, y_train, X_test):
    print("\n--- Training Neural Network (PyTorch) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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

    epochs = 10
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Vorhersage
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device))
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_processed_data()
    print(f"Data formats: Train {X_train.shape}, Test {X_test.shape}")

    xgb_preds, xgb_probs = train_xgboost(X_train, y_train, X_test)
    rf_preds, rf_probs = train_random_forest(X_train, y_train, X_test)
    nn_preds, nn_probs = train_neural_network(X_train, y_train, X_test)
    
    models = {
        "XGBoost": (xgb_preds, xgb_probs),
        "Random Forest": (rf_preds, rf_probs),
        "Neural Network": (nn_preds, nn_probs)
    }

    print("\n===========================================")
    print("       FINAL BASELINE RESULTS  ")
    print("===========================================")

    for name, (preds, probs) in models.items():
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        print(f"\n>> {name}:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC:      {auc:.4f}")