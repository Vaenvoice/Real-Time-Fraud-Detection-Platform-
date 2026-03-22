from app.core.database import engine, Base
from app.models.db_models import TransactionRecord

print("Initializing database tables...")
Base.metadata.create_all(bind=engine)
print("Database initialized successfully.")
