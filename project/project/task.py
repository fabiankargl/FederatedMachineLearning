import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo

def get_processed_data_xgb():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    y = y.copy()
    y['income'] = y["income"].astype(str).str.replace('.', '', regex=False)
    y_bin = y['income'].apply(lambda x: 0 if '<=50K' in x else 1).values

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']

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
    X_processed = preprocessor.fit_transform(X)
    return train_test_split(X_processed, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

def load_tabular_data(X_train, y_train, X_test, y_test, batch_size=32):
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

def train_tabular_xgb(model, trainloader, epochs=1):
    # XGBoost needs numpy arrays
    X = torch.cat([X for X, _ in trainloader]).numpy()
    y = torch.cat([y for _, y in trainloader]).numpy()
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X, y)
    return model

def test_tabular_xgb(model, testloader):
    X = torch.cat([X for X, _ in testloader]).numpy()
    y = torch.cat([y for _, y in testloader]).numpy()
    y_pred = model.predict(X)
    acc = (y_pred == y).mean()
    # simple logloss
    from sklearn.metrics import log_loss
    loss = log_loss(y, model.predict_proba(X))
    return loss, acc
