from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Time of the transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    Amount: float = Field(..., ge=0, description="Transaction amount")


class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: int
    result: str