import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def explain_model(processed_data_path, model_path, output_dir):
    print(f"Loading data and model for explanation...")
    X_train, X_test, y_train, y_test = joblib.load(processed_data_path)
    model = joblib.load(model_path)
    scaler = joblib.load("data/processed/scaler.joblib")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get feature names
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour', 'Log_Amount', 'Amount_to_Mean_Ratio', 'V17_V14', 'V12_V10']
    
    # 1. Logistic Regression Coefficients
    coeffs = model.coef_[0]
    importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coeffs})
    importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
    importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 12))
    sns.barplot(x='Coefficient', y='Feature', data=importance_df.head(20), palette='RdBu')
    plt.title('Top 20 Features by LogReg Coefficients')
    plt.savefig(os.path.join(output_dir, 'feature_importance_logreg.png'))
    plt.close()
    
    # 2. SHAP (on a subset for speed)
    print("\nCalculating SHAP values (this may take a moment)...")
    # Using a subset of X_test for SHAP (linear explainer is fast for LogReg)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test_df.head(500))
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df.head(500), show=False)
    plt.title('SHAP Summary Plot (Subset)')
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Explainability complete. Artifacts saved in {output_dir}")

if __name__ == "__main__":
    explain_model("data/processed/processed_data.pkl", "app/models/artifacts/best_model.joblib", "reports/explainability")
