from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from app.core.database import Base
import datetime

class TransactionRecord(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    amount = Column(Float)
    fraud_probability = Column(Float)
    fraud_label = Column(Integer)
    explanation = Column(JSON)
    status = Column(String, default="processed")
