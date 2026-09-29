from pathlib import Path
import pandas as pd

# Define Base Path relative to this file location
BASE_DIR = Path(__file__).resolve().parent.parent


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load the fraud detection dataset from a CSV file.
    """
    df = pd.read_csv(path)
    return df


def basic_dataset_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    """
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Shape: {df.shape}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))

    print(f"\nDuplicate rows: {df.duplicated().sum()}")


if __name__ == "__main__":
    DATA_PATH = BASE_DIR / "data" / "raw" / "credit_card_fraud.csv"
    if DATA_PATH.exists():
        df = load_data(DATA_PATH)
        basic_dataset_info(df)
    else:
        print(f"Data path not found: {DATA_PATH}")