from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import TransactionInput, PredictionResponse
from api.database import init_database, save_transaction, get_transactions
from api.model_service import load_model, load_scaler, predict_transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load ML model & scaler on startup."""
    init_database()
    try:
        load_model()
        load_scaler()
        print("Model and Scaler loaded successfully.")
    except FileNotFoundError as error:
        print(f"Warning: {error}")
    yield


app = FastAPI(
    title="AI Fraud Detection API",
    description="Backend API for predicting credit card fraud transactions and tracking history.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "AI Fraud Detection API is running", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        prediction, result = predict_transaction(transaction)
        transaction_id = save_transaction(transaction, prediction, result)

        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "result": result,
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(error)}")


@app.get("/transactions")
def transactions():
    try:
        return get_transactions()
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Could not retrieve transactions: {str(error)}"
        )

