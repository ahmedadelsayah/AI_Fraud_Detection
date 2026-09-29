import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET = "Class"


def basic_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("BASIC DATASET ANALYSIS")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\n--- Missing Values ---")
    print(df.isnull().sum().sort_values(ascending=False))
    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())


def target_analysis(df: pd.DataFrame, save_dir: Path = BASE_DIR / "reports") -> None:
    print("\n" + "=" * 60)
    print("TARGET ANALYSIS")
    print("=" * 60)

    if TARGET not in df.columns:
        print(f"Target column '{TARGET}' not found in dataframe.")
        return

    counts = df[TARGET].value_counts()
    percentages = df[TARGET].value_counts(normalize=True).mul(100).round(2)

    print("\nCounts:")
    print(counts)
    print("\nPercentages:")
    print(percentages)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Fraud Label")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(save_dir / "target_distribution.png")
    plt.close()


def numerical_analysis(df: pd.DataFrame, save_dir: Path = BASE_DIR / "reports") -> None:
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

    print("\n" + "=" * 60)
    print("NUMERICAL FEATURES")
    print("=" * 60)

    if len(numerical_columns) == 0:
        print("No numerical features found.")
        return

    print(df[numerical_columns].describe())

    correlation = df[numerical_columns].corr()

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Numerical Features Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_matrix.png")
    plt.close()


def categorical_analysis(df: pd.DataFrame) -> None:
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURES")
    print("=" * 60)

    if len(categorical_columns) == 0:
        print("No categorical features found.")
        return

    for column in categorical_columns:
        print(f"\n--- {column} ---")
        print(df[column].value_counts(dropna=False).head(10))


def amount_analysis(df: pd.DataFrame, save_dir: Path = BASE_DIR / "reports") -> None:
    if "Amount" not in df.columns:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Amount", bins=50, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_dir / "transaction_amount_distribution.png")
    plt.close()


def run_eda(df: pd.DataFrame, save_dir: Path = BASE_DIR / "reports") -> None:
    basic_analysis(df)
    target_analysis(df, save_dir=save_dir)
    numerical_analysis(df, save_dir=save_dir)
    categorical_analysis(df)
    amount_analysis(df, save_dir=save_dir)
    print("\nEDA completed successfully. Visualizations saved to 'reports/'.")