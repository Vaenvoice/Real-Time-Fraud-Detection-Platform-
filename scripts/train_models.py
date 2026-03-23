import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV

def train_and_compare(processed_data_path, models_dir):
    print(f"Loading processed data from {processed_data_path}...")
    X_train, X_test, y_train, y_test = joblib.load(processed_data_path)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = []
    best_recall = 0
    best_model_name = ""
    best_model = None
    
    param_grids = {
        "Random Forest": {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        "XGBoost": {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name in param_grids:
            print(f"Optimizing {name} with GridSearchCV...")
            scorer = make_scorer(recall_score)
            grid_search = GridSearchCV(model, param_grids[name], scoring=scorer, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best Params: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} Results - Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
        results.append({"Model": name, "Recall": recall, "ROC-AUC": roc_auc})
        
        # Save each model
        joblib.dump(model, os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.joblib"))
        
        if recall >= best_recall:
            best_recall = recall
            best_model_name = name
            best_model = model

    print(f"\nBest Model by Recall: {best_model_name} ({best_recall:.4f})")
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.joblib'))
    
    # Save results summary
    pd.DataFrame(results).to_csv(os.path.join(models_dir, 'model_comparison.csv'), index=False)
    print(f"Comparison results saved in {models_dir}")

if __name__ == "__main__":
    train_and_compare("data/processed/processed_data.pkl", "app/models/artifacts")
