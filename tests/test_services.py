import pytest
import os
from app.services.prediction_service import PredictionService
from app.schemas.transaction import TransactionInput

@pytest.fixture
def prediction_service():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "app", "models", "artifacts", "best_model.joblib")
    scaler_path = os.path.join(project_root, "data", "processed", "scaler.joblib")
    return PredictionService(model_path, scaler_path)

def test_engineer_features(prediction_service):
    sample_data = {
        "Time": 3600,
        "Amount": 100.0,
        "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
        "V6": 0.6, "V7": 0.7, "V8": 0.8, "V9": 0.9, "V10": 1.0,
        "V11": 1.1, "V12": 1.2, "V13": 1.3, "V14": 1.4, "V15": 1.5,
        "V16": 1.6, "V17": 1.7, "V18": 1.8, "V19": 1.9, "V20": 2.0,
        "V21": 2.1, "V22": 2.2, "V23": 2.3, "V24": 2.4, "V25": 2.5,
        "V26": 2.6, "V27": 2.7, "V28": 2.8
    }
    X_scaled = prediction_service._engineer_features(sample_data)
    assert X_scaled.shape == (1, 34) # 28 Vs + Hour + Log_Amount + Ratio + 2 interaction terms = 33? wait
    # V1-V28 (28) + Hour (1) + Log_Amount (1) + Amount_to_Mean_Ratio (1) + V17_V14 (1) + V12_V10 (1) = 33 filters + ?
    # Let's check service: self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour', 'Log_Amount', 'Amount_to_Mean_Ratio', 'V17_V14', 'V12_V10']
    # 28 + 1 (Amount) + 5 = 34. Correct.

def test_prediction_output(prediction_service):
    transaction = TransactionInput(
        Time=100.0,
        Amount=50.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    )
    result = prediction_service.predict(transaction)
    assert 0 <= result.fraud_probability <= 1
    assert result.fraud_label in [0, 1]
    assert isinstance(result.explanation, dict)
