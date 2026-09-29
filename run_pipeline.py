from pathlib import Path
from src.data_loader import load_data, basic_dataset_info
from src.eda import run_eda
from src.preprocessing import preprocess_data
import subprocess

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "credit_card_fraud.csv"


def main():
    print("\n" + "=" * 70)
    print("AI FRAUD DETECTION SYSTEM - COMPLETE PIPELINE")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"\n[ERROR] Raw dataset not found at expected path: {DATA_PATH}")
        print("Please place 'credit_card_fraud.csv' inside 'data/raw/' directory.\n")
        return

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Basic info
    basic_dataset_info(df)

    # 3. EDA
    print("\n\nRunning EDA...")
    run_eda(df)

    # 4. Preprocessing
    print("\n\nRunning Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 5. Model Training & Evaluation
    print("\n\nRunning Model Training & Evaluation...")
    data_model_path = BASE_DIR / "src" / "data_model.py"
    subprocess.run(["python", str(data_model_path)], check=True)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()