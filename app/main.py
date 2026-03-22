from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from app.schemas.transaction import TransactionInput, PredictionResponse, BatchPredictionResponse
from app.services.prediction_service import PredictionService
from app.core.logging_config import logger
from app.core.database import get_db
from app.models.db_models import TransactionRecord
import os
from typing import List

app = FastAPI(title="Real-Time Fraud Detection API", version="1.0.0")

# Dependency for PredictionService
def get_prediction_service():
    # Use absolute project root for reliability
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "app", "models", "artifacts", "best_model.joblib")
    scaler_path = os.path.join(project_root, "data", "processed", "scaler.joblib")
    
    # Check if files exist before trying to load
    if not os.path.exists(model_path):
        logger.error(f"Model file Not Found: {model_path}")
        raise FileNotFoundError(f"Missing model at {model_path}")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file Not Found: {scaler_path}")
        raise FileNotFoundError(f"Missing scaler at {scaler_path}")
        
    return PredictionService(model_path, scaler_path)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "logistic_regression", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    transaction: TransactionInput, 
    service: PredictionService = Depends(get_prediction_service),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Received prediction request for amount: {transaction.Amount}")
        result = service.predict(transaction)
        
        # PERSIST TO DATABASE
        db_record = TransactionRecord(
            amount=transaction.Amount,
            fraud_probability=result.fraud_probability,
            fraud_label=result.fraud_label,
            explanation=result.explanation
        )
        db.add(db_record)
        db.commit()
        
        if result.fraud_label == 1:
            logger.warning(f"FRAUD DETECTED! Probability: {result.fraud_probability:.4f}")
        else:
            logger.info(f"Transaction approved. Probability: {result.fraud_probability:.4f}")
            
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(transactions: List[TransactionInput], service: PredictionService = Depends(get_prediction_service)):
    try:
        logger.info(f"Processing batch of {len(transactions)} transactions")
        results = [service.predict(t) for t in transactions]
        return BatchPredictionResponse(predictions=results, total_count=len(transactions))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during batch prediction")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
