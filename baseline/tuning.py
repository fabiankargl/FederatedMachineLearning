import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
# Importiere deine Daten-Funktion aus centralized_benchmark.py
# (Stelle sicher, dass centralized_benchmark.py im selben Ordner liegt)
from centralized_baseline import get_processed_data 

# Parameter-Gitter (wie besprochen)
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 6, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 3, 5] 
}

param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

def save_results(search_object, filename):
    """Hilfsfunktion: Speichert alle Ergebnisse des Tunings in eine CSV."""
    print(f"   -> Speichere Ergebnisse in '{filename}'...")
    
    # cv_results_ ist ein Dictionary mit allen Details
    results = pd.DataFrame(search_object.cv_results_)
    
    # Wir filtern die Tabelle, damit sie lesbarer ist (nur Params & Scores)
    # Spalten, die mit 'param_' oder 'mean_test' beginnen + std_test
    cols_to_keep = [col for col in results.columns if 'param_' in col or 'mean_test' in col or 'std_test' in col]
    # Wichtig: Rank hinzufügen
    cols_to_keep.append('rank_test_f1')
    
    # Speichern (sortiert nach bestem F1 Score)
    results_filtered = results[cols_to_keep].sort_values(by='rank_test_f1')
    results_filtered.to_csv(filename, index=False)
    print("      Fertig.")

def run_tuning():
    print("--- Lade Daten für Tuning ---")
    X_train, _, y_train, _ = get_processed_data() 

    # Cross-Validation Strategie
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Wir wollen BEIDE Metriken tracken
    scoring_metrics = {'f1': 'f1', 'auc': 'roc_auc'}

    # -------------------------------------------------------
    # 1. XGBoost Tuning
    # -------------------------------------------------------
    print("\n--- Starte XGBoost Tuning... ---")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist' 
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist_xgb, 
        n_iter=20, # Anzahl der Versuche
        scoring=scoring_metrics, # Hier übergeben wir beide Metriken
        refit='f1',              # Welches ist das "Haupt"-Ziel für den besten Estimator?
        n_jobs=-1, 
        cv=cv, 
        verbose=1,
        random_state=42,
        return_train_score=False
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"Beste XGBoost Params: {xgb_search.best_params_}")
    print(f"Bester XGBoost F1:    {xgb_search.best_score_:.4f}")
    # Den AUC des besten Modells manuell aus dem Ergebnis holen:
    best_idx = xgb_search.best_index_
    best_auc = xgb_search.cv_results_['mean_test_auc'][best_idx]
    print(f"Zugehöriger AUC:      {best_auc:.4f}")
    
    # Speichern
    save_results(xgb_search, "tuning_results_xgboost.csv")


    # -------------------------------------------------------
    # 2. Random Forest Tuning
    # -------------------------------------------------------
    print("\n--- Starte Random Forest Tuning... ---")
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
    
    print(f"Beste RF Params: {rf_search.best_params_}")
    print(f"Bester RF F1:      {rf_search.best_score_:.4f}")
    
    best_idx_rf = rf_search.best_index_
    best_auc_rf = rf_search.cv_results_['mean_test_auc'][best_idx_rf]
    print(f"Zugehöriger AUC:   {best_auc_rf:.4f}")

    # Speichern
    save_results(rf_search, "tuning_results_rf.csv")

if __name__ == "__main__":
    run_tuning()