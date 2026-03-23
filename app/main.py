from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import func
from app.schemas.transaction import TransactionInput, PredictionResponse, BatchPredictionResponse, DashboardStats, TransactionLog
from app.services.prediction_service import PredictionService
from app.core.logging_config import logger
from app.core.database import get_db
from app.models.db_models import TransactionRecord
import os
from typing import List

app = FastAPI(title="Real-Time Fraud Detection Engine", version="1.0.0")

# PROJECT ROOT
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/stats", response_model=DashboardStats)
async def get_stats(db: Session = Depends(get_db)):
    try:
        total = db.query(func.count(TransactionRecord.id)).scalar() or 0
        fraud = db.query(func.count(TransactionRecord.id)).filter(TransactionRecord.fraud_label == 1).scalar() or 0
        avg_risk = db.query(func.avg(TransactionRecord.fraud_probability)).scalar() or 0.0
        
        # Recent fraud rate (last 100 transactions)
        recent_txs = db.query(TransactionRecord.fraud_label).order_by(TransactionRecord.timestamp.desc()).limit(100).all()
        recent_fraud_rate = sum([tx.fraud_label for tx in recent_txs]) / len(recent_txs) if recent_txs else 0.0
        
        return DashboardStats(
            total_scanned=total,
            fraud_detected=fraud,
            avg_risk_score=float(avg_risk),
            recent_fraud_rate=float(recent_fraud_rate)
        )
    except Exception as e:
        logger.error(f"Failed to fetch stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching dashboard stats")

@app.get("/recent-transactions", response_model=List[TransactionLog])
async def get_recent_transactions(limit: int = 15, db: Session = Depends(get_db)):
    try:
        transactions = db.query(TransactionRecord).order_by(TransactionRecord.timestamp.desc()).limit(limit).all()
        # Convert timestamp to string for JSON serialization
        results = []
        for tx in transactions:
            results.append(TransactionLog(
                id=tx.id,
                timestamp=tx.timestamp.isoformat() + "Z",
                amount=tx.amount,
                fraud_probability=tx.fraud_probability,
                fraud_label=tx.fraud_label,
                explanation=tx.explanation
            ))
        return results
    except Exception as e:
        logger.error(f"Failed to fetch recent transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching recent transactions")

# SERVE FRONTEND ASSETS
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "S.H.I.E.L.D API is running. Frontend not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
