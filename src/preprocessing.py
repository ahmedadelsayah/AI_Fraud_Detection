import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from pathlib import Path


# =========================
# Settings
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "Amount"
]

TARGET = "Class"


# =========================
# Preprocessing Function
# =========================

def preprocess_data(df):

    print("\nOriginal shape:", df.shape)

    # =========================
    # Keep only required columns
    # =========================

    df = df[FEATURES + [TARGET]]

    print("\nSelected columns:")
    print(df.columns.tolist())

    # =========================
    # Check missing values
    # =========================

    print("\nMissing values:")
    print(df.isnull().sum())

    # =========================
    # Remove duplicates
    # =========================

    df = df.drop_duplicates()

    print("\nShape after duplicates:", df.shape)

    # =========================
    # Remove missing values
    # =========================

    df = df.dropna()

    print("\nShape after removing missing values:", df.shape)

    # =========================
    # X and y
    # =========================

    X = df[FEATURES]
    y = df[TARGET]

    # =========================
    # Check target
    # =========================

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nClass percentage:")
    print(y.value_counts(normalize=True) * 100)

    # =========================
    # Train / Test Split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # =========================
    # Scaling
    # =========================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, OUTPUT_PATH / "scaler.pkl")

    # =========================
    # SMOTE
    # =========================

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(X_train,y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train).value_counts())

    # =========================
    # Save processed data
    # =========================
    pd.DataFrame(X_train,columns=FEATURES).to_csv(OUTPUT_PATH / "X_train.csv",index=False)
    pd.DataFrame(X_test,columns=FEATURES).to_csv(OUTPUT_PATH / "X_test.csv",index=False)
    pd.DataFrame(y_train,columns=[TARGET]).to_csv(OUTPUT_PATH / "y_train.csv",index=False)
    pd.DataFrame(y_test,columns=[TARGET]).to_csv(OUTPUT_PATH / "y_test.csv",index=False)

    print("\nPreprocessing completed successfully!")

    return X_train, X_test, y_train, y_test