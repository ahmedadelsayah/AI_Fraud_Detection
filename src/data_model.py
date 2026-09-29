import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ----------------------------------------------------
# 1. Load the preprocessed data
# ----------------------------------------------------
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ----------------------------------------------------
# 2. Define the three baseline models
# ----------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        random_state=42, eval_metric="logloss", n_jobs=-1
    ),
}

# ----------------------------------------------------
# 3. Train and evaluate each model, collect results
# ----------------------------------------------------
results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Precision (Fraud)": round(precision_score(y_test, y_pred), 4),
        "Recall (Fraud)": round(recall_score(y_test, y_pred), 4),
        "F1-Score (Fraud)": round(f1_score(y_test, y_pred), 4),
        "AUC-ROC": round(roc_auc_score(y_test, y_proba), 4),
    })

# ----------------------------------------------------
# 4. Final comparison table
# ----------------------------------------------------
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values(by="AUC-ROC", ascending=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_path = PROCESSED_DIR / "model_comparison.csv"
comparison_df.to_csv(comparison_path,index=False)

print(f"\nComparison table saved to: {comparison_path}")

# ----------------------------------------------------
# 5. Hyperparameter tuning for the best model (by AUC-ROC)
# ----------------------------------------------------
best_model_name = comparison_df.iloc[0]["Model"]
print("\n" + "=" * 60)
print(f"Best model based on AUC-ROC: {best_model_name}")
print("Starting Hyperparameter Tuning (RandomizedSearchCV)...")
print("This may take several minutes depending on your machine.")
print("=" * 60)

if best_model_name == "Random Forest":
    tuning_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
elif best_model_name == "XGBoost":
    tuning_estimator = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
else:  # Logistic Regression fallback
    tuning_estimator = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    }

search = RandomizedSearchCV(
    estimator=tuning_estimator,
    param_distributions=param_grid,
    n_iter=15,
    scoring="f1",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(search.best_params_)
print(f"\nBest F1-Score (Cross-Validation): {search.best_score_:.4f}")

# ----------------------------------------------------
# 6. Evaluate the tuned model on the test set
# ----------------------------------------------------
best_model = search.best_estimator_

y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print(f"FINAL TUNED MODEL ({best_model_name}) - TEST SET RESULTS")
print("=" * 60)
print(classification_report(y_test, y_pred_final))
print(f"Precision (Fraud): {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall (Fraud):    {recall_score(y_test, y_pred_final):.4f}")
print(f"F1-Score (Fraud):  {f1_score(y_test, y_pred_final):.4f}")
print(f"AUC-ROC:           {roc_auc_score(y_test, y_proba_final):.4f}")

# ----------------------------------------------------
# 7. Save the final model (ready for handoff to Section 3 - API)
# ----------------------------------------------------
joblib.dump(best_model, PROCESSED_DIR / "final_fraud_model.pkl")
print(f"\nFinal tuned model saved to: {PROCESSED_DIR / 'final_fraud_model.pkl'}")