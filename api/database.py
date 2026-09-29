from pathlib import Path
import sqlite3
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "fraud_detection.db"


def get_connection():
    """
    Create and return a SQLite database connection.
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    return connection


def init_database():
    """
    Create the transactions table if it does not already exist.
    """
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Time REAL NOT NULL,
            V1 REAL NOT NULL,
            V2 REAL NOT NULL,
            V3 REAL NOT NULL,
            V4 REAL NOT NULL,
            V5 REAL NOT NULL,
            Amount REAL NOT NULL,
            prediction INTEGER NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_transaction(transaction, prediction, result):
    """
    Save a prediction and its transaction data into SQLite.
    """
    connection = get_connection()
    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO transactions (
            Time,
            V1,
            V2,
            V3,
            V4,
            V5,
            Amount,
            prediction,
            result,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transaction.Time,
            transaction.V1,
            transaction.V2,
            transaction.V3,
            transaction.V4,
            transaction.V5,
            transaction.Amount,
            prediction,
            result,
            created_at,
        ),
    )

    transaction_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return transaction_id


def get_transactions():
    """
    Return all stored transactions.
    """
    connection = get_connection()
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]