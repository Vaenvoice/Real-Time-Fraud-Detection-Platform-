from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TransactionInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., gt=0)
    Time: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_label: int
    explanation: Dict[str, float]
    status: str = "success"

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int

class DashboardStats(BaseModel):
    total_scanned: int
    fraud_detected: int
    avg_risk_score: float
    recent_fraud_rate: float

class TransactionLog(BaseModel):
    id: int
    timestamp: str
    amount: float
    fraud_probability: float
    fraud_label: int
    explanation: Dict[str, float]

    class Config:
        from_attributes = True
