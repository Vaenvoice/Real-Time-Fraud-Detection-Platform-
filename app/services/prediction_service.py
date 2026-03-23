import joblib
import pandas as pd
import numpy as np
import os
import shap
from typing import Dict
from app.schemas.transaction import TransactionInput, PredictionResponse

class PredictionService:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        # Feature names should match training
        self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour', 'Log_Amount', 'Amount_to_Mean_Ratio', 'V17_V14', 'V12_V10']
        
        # Initialize SHAP explainer (optimized for linear models)
        self.explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(data=np.zeros((1, 34))))
        
    def _engineer_features(self, data: Dict) -> pd.DataFrame:
        # 1. Convert to DataFrame
        df = pd.DataFrame([data])
        
        # 2. Replicate script feature engineering
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Log_Amount'] = np.log1p(df['Amount'])
        # Simplified for real-time: using a global mean (from training or config)
        global_mean_amount = 88.35 # Approximate mean from creditcard dataset
        df['Amount_to_Mean_Ratio'] = df['Amount'] / (global_mean_amount + 1e-5)
        
        df['V17_V14'] = df['V17'] * df['V14']
        df['V12_V10'] = df['V12'] * df['V10']
        
        # 3. Scale
        X = df.drop(['Time'], axis=1) # Same as training
        X_scaled = self.scaler.transform(X[self.feature_names])
        return X_scaled

    def predict(self, transaction: TransactionInput) -> PredictionResponse:
        input_data = transaction.model_dump()
        X_processed = self._engineer_features(input_data)
        
        # Inference
        prob = self.model.predict_proba(X_processed)[0][1]
        label = 1 if prob > 0.5 else 0
        
        # SHAP EXPLANATION
        shap_values = self.explainer.shap_values(X_processed)
        # For binary classification, shap_values[1] or similar depending on version
        # LinearExplainer returns a single array for logreg usually
        explanations = dict(zip(self.feature_names, shap_values[0]))
        
        # Sort by impact (absolute value)
        top_explanations = dict(sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return PredictionResponse(
            fraud_probability=float(prob),
            fraud_label=int(label),
            explanation=top_explanations
        )

# Initialize singleton
# These will be updated by the main app via dependency injection
MODEL_PATH = ""
SCALER_PATH = ""
