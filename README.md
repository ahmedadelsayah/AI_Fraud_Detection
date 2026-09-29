# 🛡️ AI Fraud Detection System

An end-to-end Machine Learning system for detecting fraudulent credit card transactions.

The project covers the complete workflow from data loading and exploratory data analysis (EDA), through preprocessing and model training, to a REST API for real-time fraud prediction and transaction history management.

---

## 📌 Project Overview

The **AI Fraud Detection System** is designed to analyze credit card transaction data and classify transactions as:

- **Fraud**
- **Not Fraud**

The system consists of three main stages:

1. **Data & EDA**
2. **Machine Learning Model**
3. **API Backend**

The final trained model is integrated with a **FastAPI backend**, which receives transaction data, performs prediction, and stores the transaction history in a **SQLite database**.

---

## 📂 Project Structure

```text
AI-Fraud-Detection-System/
│
├── data/
│   ├── raw/
│   │   └── credit_card_fraud.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── scaler.pkl
│       ├── model_comparison.csv
│       └── final_fraud_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── data_model.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── model_service.py
│   └── schemas.py
│
├── database/
│   └── fraud_detection.db
│
├── reports/
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   └── transaction_amount_distribution.png
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# 🔹 1. Data Loading

The project starts by loading the raw dataset from:

```text
data/raw/credit_card_fraud.csv
```

The `data_loader.py` module provides:

- Dataset loading
- Dataset shape
- Data types
- Missing value information
- Duplicate row count

---

# 🔹 2. Exploratory Data Analysis (EDA)

The `eda.py` module performs exploratory analysis before model training.

The current dataset contains the following features:

```text
Time
V1
V2
V3
V4
V5
Amount
Class
```

### EDA includes:

- Dataset dimensions
- Missing values
- Duplicate rows
- Target class distribution
- Numerical feature statistics
- Correlation matrix
- Transaction amount distribution

### Generated Reports

The following visualizations are saved inside:

```text
reports/
```

### Target Distribution

```text
reports/target_distribution.png
```

Shows the distribution between fraudulent and non-fraudulent transactions.

### Correlation Matrix

```text
reports/correlation_matrix.png
```

Shows the correlations between numerical features.

### Transaction Amount Distribution

```text
reports/transaction_amount_distribution.png
```

Shows the distribution of transaction amounts.

---

# 🔹 3. Data Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocessing.py
```

### Selected Features

The model currently uses:

```text
Time
V1
V2
V3
V4
V5
Amount
```

Target:

```text
Class
```

Where:

```text
0 → Not Fraud
1 → Fraud
```

### Preprocessing Steps

The pipeline performs:

1. Selecting the required features
2. Checking missing values
3. Removing duplicate rows
4. Removing missing values
5. Separating features and target
6. Stratified train/test split
7. Feature scaling using `StandardScaler`
8. Handling class imbalance using `SMOTE`
9. Saving the processed datasets

The train/test split uses:

```text
80% Training
20% Testing
```

with:

```text
random_state = 42
```

The fitted scaler is saved as:

```text
data/processed/scaler.pkl
```

This scaler is reused by the API when making predictions on new transactions.

---

# 🔹 4. Machine Learning Models

The model training pipeline is implemented in:

```text
src/data_model.py
```

Three classification models are evaluated:

### Logistic Regression

A linear classification model used as a baseline.

### Random Forest

An ensemble model based on multiple decision trees.

### XGBoost

A gradient boosting model designed for high-performance classification.

---

## 📊 Model Evaluation

The models are evaluated using:

- Precision
- Recall
- F1-Score
- AUC-ROC

The comparison results are saved to:

```text
data/processed/model_comparison.csv
```

The model with the highest AUC-ROC is selected for further hyperparameter tuning.

---

# 🔹 5. Hyperparameter Tuning

After comparing the baseline models, the selected model is tuned using:

```text
RandomizedSearchCV
```

The tuning process uses:

```text
3-fold Cross Validation
```

and optimizes:

```text
F1-Score
```

The final tuned model is evaluated on the test set.

---

# 🔹 6. Final Model

The final trained model is saved as:

```text
data/processed/final_fraud_model.pkl
```

This model is used by the API backend for real-time transaction prediction.

---

# 🔹 7. API Backend

The backend is implemented using:

```text
FastAPI
```

The API is responsible for:

- Receiving transaction data
- Validating input
- Loading the trained model
- Applying the saved scaler
- Making fraud predictions
- Saving prediction results
- Providing transaction history

---

## 🚀 API Endpoints

### `GET /`

Returns a basic API status message.

Example response:

```json
{
  "message": "AI Fraud Detection API is running",
  "docs": "/docs"
}
```

---

### `GET /health`

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy"
}
```

---

### `POST /predict`

Receives transaction information and returns the fraud prediction.

#### Request

```json
{
  "Time": 1000,
  "V1": -1.2,
  "V2": 0.5,
  "V3": 1.1,
  "V4": -0.3,
  "V5": 0.8,
  "Amount": 150.5
}
```

#### Response

```json
{
  "transaction_id": 1,
  "prediction": 0,
  "result": "Not Fraud"
}
```

or:

```json
{
  "transaction_id": 2,
  "prediction": 1,
  "result": "Fraud"
}
```

---

### `GET /transactions`

Returns the stored transaction history from the SQLite database.

Example:

```json
[
  {
    "id": 1,
    "Time": 1000,
    "V1": -1.2,
    "V2": 0.5,
    "V3": 1.1,
    "V4": -0.3,
    "V5": 0.8,
    "Amount": 150.5,
    "prediction": 0,
    "result": "Not Fraud",
    "created_at": "2026-09-29T01:00:00"
  }
]
```

---

# 🔹 8. Database

The backend uses:

```text
SQLite
```

The database file is:

```text
database/fraud_detection.db
```

The `transactions` table stores:

```text
id
Time
V1
V2
V3
V4
V5
Amount
prediction
result
created_at
```

Each prediction made through:

```text
POST /predict
```

is automatically stored in the database.

---

# 🔹 9. API Documentation

FastAPI automatically generates interactive API documentation.

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface allows you to:

- View available endpoints
- View request schemas
- Test API endpoints
- Send prediction requests
- View API responses

---

# 🔹 10. Running the Project

## Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

---

## Run the Complete ML Pipeline

From:

```text
AI-Fraud-Detection-System/
```

run:

```bash
python run_pipeline.py
```

This executes:

```text
Data Loading
     ↓
EDA
     ↓
Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Final Model
```

---

# 🔹 11. Run the API

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🔹 12. Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn / SMOTE

### Model Persistence

- Joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Database

- SQLite

### Development

- Git
- GitHub
- VS Code

---

# 🎯 Project Workflow

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
EDA & Visualization
     │
     ▼
Data Cleaning
     │
     ▼
Train / Test Split
     │
     ▼
Feature Scaling
     │
     ▼
SMOTE
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     ├── Random Forest
     └── XGBoost
     │
     ▼
Model Comparison
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Final Fraud Detection Model
     │
     ▼
FastAPI Backend
     │
     ├── /predict
     ├── /transactions
     └── /health
     │
     ▼
SQLite Transaction History
```

---

## 👥 Project Sections

The system is organized into three main sections:

### Section 1 — Data

Responsible for:

- Data loading
- Data exploration
- Data cleaning
- Dataset analysis

### Section 2 — Machine Learning

Responsible for:

- Preprocessing
- Feature scaling
- SMOTE
- Model training
- Model comparison
- Hyperparameter tuning
- Final model generation

### Section 3 — API Backend

Responsible for:

- FastAPI backend
- Model integration
- Transaction prediction
- Input validation
- SQLite database
- Transaction history
- API documentation

---

## 📌 Current API Input Features

The API expects the following seven features:

```text
Time
V1
V2
V3
V4
V5
Amount
```

The feature order is kept consistent with the machine learning pipeline to ensure correct predictions.

---

## 🔐 Prediction Logic

The API follows this process for every transaction:

```text
Transaction Request
        ↓
Input Validation
        ↓
Feature Ordering
        ↓
StandardScaler
        ↓
Trained ML Model
        ↓
Prediction
        ↓
Fraud / Not Fraud
        ↓
Save Transaction
        ↓
Return JSON Response
```

