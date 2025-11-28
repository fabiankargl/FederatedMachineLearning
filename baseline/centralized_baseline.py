from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_utils import get_processed_data
from model_utils import train_xgboost, train_random_forest, train_neural_network

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_processed_data()

    xgb_preds, xgb_probs = train_xgboost(X_train, y_train, X_test)
    rf_preds, rf_probs = train_random_forest(X_train, y_train, X_test)
    nn_preds, nn_probs = train_neural_network(X_train, y_train, X_test)
    
    models = {
        "XGBoost": (xgb_preds, xgb_probs),
        "Random Forest": (rf_preds, rf_probs),
        "Neural Network": (nn_preds, nn_probs)
    }
    print("Final baseline results:")

    for name, (preds, probs) in models.items():
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        print(f"\n>> {name}:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC:      {auc:.4f}")