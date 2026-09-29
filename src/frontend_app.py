"""
AI Fraud Detection System - Frontend Dashboard (Streamlit)

Talks to the FastAPI backend (api/main.py) over HTTP.
Run the API first:
    uvicorn api.main:app --reload

Then run this dashboard:
    streamlit run frontend/app.py
"""

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

st.title("💳 AI Fraud Detection System")
st.caption("Predict whether a credit card transaction is fraudulent, and review prediction history.")

tab_predict, tab_history, tab_about = st.tabs(["🔍 Check a Transaction", "📊 Transaction History", "ℹ️ About"])

# ----------------------------------------------------------------
# Tab 1: Predict a single transaction
# ----------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Transaction Details")
    st.write(
        "V1–V5 are anonymized (PCA-transformed) features from the original dataset. "
        "If you're testing with a real row from the dataset, paste the values directly."
    )

    col1, col2 = st.columns(2)

    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
        v1 = st.number_input("V1", value=0.0, format="%.6f")
        v2 = st.number_input("V2", value=0.0, format="%.6f")
        v3 = st.number_input("V3", value=0.0, format="%.6f")

    with col2:
        v4 = st.number_input("V4", value=0.0, format="%.6f")
        v5 = st.number_input("V5", value=0.0, format="%.6f")
        amount = st.number_input("Amount ($)", value=0.0, min_value=0.0, step=1.0)

    if st.button("🔍 Check Transaction", type="primary", use_container_width=True):
        payload = {
            "Time": time_val,
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "V5": v5,
            "Amount": amount,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result["result"] == "Fraud":
                st.error(f"🚨 **FRAUD DETECTED** — Transaction #{result['transaction_id']}")
            else:
                st.success(f"✅ **Not Fraud** — Transaction #{result['transaction_id']}")

            st.json(result)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the API. Make sure it's running:\n\n"
                "`uvicorn api.main:app --reload`"
            )
        except requests.exceptions.HTTPError as error:
            st.error(f"API error: {error}")
        except Exception as error:
            st.error(f"Unexpected error: {error}")

# ----------------------------------------------------------------
# Tab 2: Transaction history + dashboard
# ----------------------------------------------------------------
with tab_history:
    st.subheader("Prediction History")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/transactions", timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            st.info("No transactions recorded yet. Try checking one in the first tab.")
        else:
            df = pd.DataFrame(data)

            # ---- Summary metrics ----
            total = len(df)
            fraud_count = (df["result"] == "Fraud").sum()
            fraud_rate = (fraud_count / total * 100) if total else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Transactions Checked", total)
            m2.metric("Flagged as Fraud", int(fraud_count))
            m3.metric("Fraud Rate", f"{fraud_rate:.1f}%")

            # ---- Charts ----
            c1, c2 = st.columns(2)

            with c1:
                counts = df["result"].value_counts().reset_index()
                counts.columns = ["Result", "Count"]
                fig = px.pie(counts, names="Result", values="Count", title="Fraud vs Not Fraud")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig2 = px.histogram(df, x="Amount", color="result", title="Amount Distribution by Result", nbins=30)
                st.plotly_chart(fig2, use_container_width=True)

            # ---- Table ----
            st.subheader("All Transactions")
            st.dataframe(df, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to the API. Make sure it's running:\n\n"
            "`uvicorn api.main:app --reload`"
        )
    except Exception as error:
        st.error(f"Unexpected error: {error}")

# ----------------------------------------------------------------
# Tab 3: About
# ----------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **AI Fraud Detection System** — a graduation project pipeline covering:

        1. **Data & EDA** — Kaggle Credit Card Fraud dataset, cleaned and explored
        2. **Modeling** — trained and compared multiple classifiers
        3. **API (Backend)** — FastAPI service serving real-time predictions and storing history in SQLite
        4. **Frontend (this dashboard)** — Streamlit interface for testing transactions and reviewing history

        The model currently used by the API is trained on 7 features: `Time`, `V1`–`V5`, and `Amount`.
        """
    )

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("✅ API is online and reachable.")
        else:
            st.warning("⚠️ API responded but with an unexpected status.")
    except requests.exceptions.ConnectionError:
        st.error("❌ API is not reachable. Start it with: `uvicorn api.main:app --reload`")
