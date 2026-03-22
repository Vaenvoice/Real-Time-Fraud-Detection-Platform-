import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score

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
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} Results - Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
        results.append({"Model": name, "Recall": recall, "ROC-AUC": roc_auc})
        
        # Save each model
        joblib.dump(model, os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.joblib"))
        
        if recall > best_recall:
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
