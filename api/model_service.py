from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "data" / "processed" / "final_fraud_model.pkl"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"


FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "Amount",
]


model = None
scaler = None


def load_model():
    """
    Load the trained fraud detection model.
    """
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


def load_scaler():
    """
    Load the StandardScaler used during training.
    """
    global scaler

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    return scaler


def predict_transaction(transaction):
    """
    Preprocess a transaction and make a fraud prediction.
    """

    global model, scaler

    if model is None:
        load_model()

    if scaler is None:
        load_scaler()

    # Convert input to DataFrame
    data = pd.DataFrame(
        [
            {
                "Time": transaction.Time,
                "V1": transaction.V1,
                "V2": transaction.V2,
                "V3": transaction.V3,
                "V4": transaction.V4,
                "V5": transaction.V5,
                "Amount": transaction.Amount,
            }
        ]
    )

    # Make sure the feature order is exactly the same
    data = data[FEATURES]

    # Apply the same scaler used during training
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = int(model.predict(data_scaled)[0])

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Not Fraud"

    return prediction, result