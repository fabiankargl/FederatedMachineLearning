import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from baseline.data_utils import get_processed_data
from typing import Dict, Any, List

param_dist_xgb: Dict[str, List[Any]] = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 6, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 3, 5] 
}

param_dist_rf: Dict[str, List[Any]] = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

def save_results(search_object: RandomizedSearchCV, 
                 filename: str) -> None:
    """
    Saves the results of a RandomizedSearchCV to a CSV file.

    Args:
        search_object (RandomizedSearchCV): The fitted search object.
        filename (str): The name of the file to save the results to.
    """
    print(f"-> Save results in '{filename}'")
    
    results = pd.DataFrame(search_object.cv_results_)
    
    cols_to_keep = [col for col in results.columns if 'param_' in col or 'mean_test' in col or 'std_test' in col]
    cols_to_keep.append('rank_test_f1')
    
    results_filtered = results[cols_to_keep].sort_values(by='rank_test_f1')
    results_filtered.to_csv(filename, index=False)
    print("Finished")

def run_tuning() -> None:
    """
    Performs hyperparameter tuning for XGBoost and Random Forest models
    using RandomizedSearchCV on the preprocessed Adult dataset.
    """
    X_train, _, y_train, _ = get_processed_data() 

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scoring_metrics = {'f1': 'f1', 'auc': 'roc_auc'}

    print("\n--- Start XGBoost tuning ---")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist' 
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist_xgb, 
        n_iter=20, 
        scoring=scoring_metrics, 
        refit='f1',              
        n_jobs=-1, 
        cv=cv, 
        verbose=1,
        random_state=42,
        return_train_score=False
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"Best XGBoost Params: {xgb_search.best_params_}")
    print(f"Best XGBoost F1: {xgb_search.best_score_:.4f}")
    best_idx = xgb_search.best_index_
    best_auc = xgb_search.cv_results_['mean_test_auc'][best_idx]
    print(f"Best XGBoost AUC: {best_auc:.4f}")
    
    save_results(xgb_search, "tuning_results_xgboost.csv")

    print("\n--- Start Random Forest tuning ---")
    rf_model = RandomForestClassifier(random_state=42)
    
    rf_search = RandomizedSearchCV(
        rf_model, 
        param_distributions=param_dist_rf, 
        n_iter=20, 
        scoring=scoring_metrics, 
        refit='f1', 
        n_jobs=-1, 
        cv=cv, 
        verbose=1,
        random_state=42,
        return_train_score=False
    )
    
    rf_search.fit(X_train, y_train)
    
    print(f"Best RF Params: {rf_search.best_params_}")
    print(f"Best RF F1: {rf_search.best_score_:.4f}")
    
    best_idx_rf = rf_search.best_index_
    best_auc_rf = rf_search.cv_results_['mean_test_auc'][best_idx_rf]
    print(f"Best RF AUC: {best_auc_rf:.4f}")

    save_results(rf_search, "tuning_results_rf.csv")

if __name__ == "__main__":
    run_tuning()