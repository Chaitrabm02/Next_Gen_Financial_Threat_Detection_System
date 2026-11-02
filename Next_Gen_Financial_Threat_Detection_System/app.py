import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_utils import prepare_dataframe, generate_demo_data, convert_to_inr
from src.train import train_models, load_model_bundle, predict_dataframe, MODEL_PATH

st.set_page_config(page_title="Next Generation Financial Threat Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Next Generation Financial Threat Detection System")
st.caption("Machine-learning-based transaction screening | Academic demonstration prototype")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a module", [
    "Dashboard", "Single Transaction Check", "Batch Fraud Detection",
    "Train / Evaluate Model", "Dataset Explorer", "About"
])

@st.cache_data
def demo_data():
    p = Path("data/demo_transactions.csv")
    if p.exists():
        return pd.read_csv(p)
    d = generate_demo_data(5000)
    p.parent.mkdir(exist_ok=True)
    d.to_csv(p, index=False)
    return d

def get_bundle():
    try:
        return load_model_bundle()
    except Exception:
        return None

if "data" not in st.session_state:
    st.session_state.data = demo_data()

df = st.session_state.data.copy()
if page == "Dashboard":
    st.subheader("Transaction monitoring dashboard")
    bundle = get_bundle()
    if bundle is not None:
        result = predict_dataframe(df, bundle)
        fraud_col = result["prediction"]
        fraud_count = int((fraud_col == 1).sum())
        total = len(result)
        rate = fraud_count / total * 100 if total else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions analysed", f"{total:,}")
        c2.metric("Predicted fraud", f"{fraud_count:,}")
        c3.metric("Predicted legitimate", f"{total-fraud_count:,}")
        c4.metric("Predicted fraud rate", f"{rate:.2f}%")
        left, right = st.columns(2)
        with left:
            st.markdown("**Predicted transaction distribution**")
            counts = result["prediction"].map({0:"Legitimate",1:"Fraud"}).value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Transactions")
            ax.set_xlabel("")
            st.pyplot(fig)
        with right:
            st.markdown("**Transaction amount distribution (INR)**")
            fig, ax = plt.subplots()
            sns.histplot(result["amount_inr"], bins=30, ax=ax)
            ax.set_xlabel("Amount (INR)")
            st.pyplot(fig)
        st.markdown("**Highest-risk transactions**")
        show = result.sort_values("fraud_probability", ascending=False).head(15)
        st.dataframe(show, use_container_width=True)
    else:
        st.warning("No trained model found. Open **Train / Evaluate Model** and train one.")
        st.info("The project includes a demo dataset. Train a model to populate this dashboard.")

elif page == "Single Transaction Check":
    st.subheader("Check an individual transaction")
    bundle = get_bundle()
    if bundle is None:
        st.warning("Train a model first using the Train / Evaluate Model page.")
    else:
        with st.form("transaction_form"):
            c1,c2,c3 = st.columns(3)
            amount = c1.number_input("Transaction amount", min_value=0.01, value=2500.0, step=100.0)
            currency = c2.selectbox("Currency", ["INR","USD","EUR","GBP","AED","SGD"])
            txn_type = c3.selectbox("Transaction type", ["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"])
            hour = c1.slider("Transaction hour", 0, 23, 14)
            oldbalance = c2.number_input("Sender balance before transaction", min_value=0.0, value=10000.0)
            newbalance = c3.number_input("Sender balance after transaction", min_value=0.0, value=7500.0)
            recipient_old = c1.number_input("Recipient balance before transaction", min_value=0.0, value=5000.0)
            recipient_new = c2.number_input("Recipient balance after transaction", min_value=0.0, value=7500.0)
            location = c3.selectbox("Location category", ["Usual","New","International"])
            submitted = st.form_submit_button("Analyse transaction", type="primary")
        if submitted:
            one = pd.DataFrame([{
                "amount": amount, "currency": currency, "type": txn_type, "hour": hour,
                "oldbalanceOrg": oldbalance, "newbalanceOrig": newbalance,
                "oldbalanceDest": recipient_old, "newbalanceDest": recipient_new,
                "location": location
            }])
            result = predict_dataframe(one, bundle)
            row = result.iloc[0]
            if int(row["prediction"]) == 1:
                st.error("⚠️ HIGH RISK: This transaction is classified as potentially fraudulent.")
            else:
                st.success("✅ LOW RISK: This transaction is classified as likely legitimate.")
            st.metric("Estimated fraud probability", f"{row['fraud_probability']*100:.2f}%")
            st.metric("Amount converted to INR", f"₹{row['amount_inr']:,.2f}")
            st.caption("This is a model estimate, not a bank authorization or proof of fraud.")

elif page == "Batch Fraud Detection":
    st.subheader("Upload and screen multiple transactions")
    st.write("Upload a CSV with transaction columns. A sample CSV is available in `data/demo_transactions.csv`.")
    uploaded = st.file_uploader("Choose transaction CSV", type=["csv"])
    bundle = get_bundle()
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            st.write("Preview", batch.head())
            if bundle is None:
                st.warning("Train a model first.")
            elif st.button("Run batch detection", type="primary"):
                result = predict_dataframe(batch, bundle)
                st.session_state.batch_result = result
                st.success(f"Analysed {len(result)} transactions.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    if "batch_result" in st.session_state:
        result = st.session_state.batch_result
        st.dataframe(result, use_container_width=True)
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Download fraud screening report", csv, "fraud_detection_report.csv", "text/csv")
        frauds = result[result.prediction == 1]
        st.metric("Flagged transactions", len(frauds))
        st.download_button("Download flagged transactions only",
                           frauds.to_csv(index=False).encode("utf-8"),
                           "flagged_transactions.csv", "text/csv")

elif page == "Train / Evaluate Model":
    st.subheader("Train and compare machine learning models")
    st.write("The application compares Logistic Regression and Random Forest using a stratified hold-out test set.")
    uploaded = st.file_uploader("Optional: upload labelled training CSV", type=["csv"], key="train_upload")
    if uploaded is not None:
        try:
            train_df = pd.read_csv(uploaded)
            st.session_state.training_data = train_df
            st.success(f"Loaded {len(train_df):,} rows.")
        except Exception as e:
            st.error(f"Invalid CSV: {e}")
    use_uploaded = "training_data" in st.session_state
    if st.button("Train models", type="primary"):
        train_df = st.session_state.training_data if use_uploaded else df
        with st.spinner("Training models..."):
            try:
                metrics = train_models(train_df)
                st.session_state.metrics = metrics
                st.success("Training completed. Best model saved to models/model_bundle.joblib.")
            except Exception as e:
                st.exception(e)
    if "metrics" in st.session_state:
        metrics = st.session_state.metrics
        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
        best = max(metrics, key=lambda k: metrics[k]["f1"])
        st.info(f"Selected model: {best} (highest F1-score).")
        st.caption("For fraud detection, precision, recall, F1-score and ROC-AUC are more informative than accuracy alone.")

elif page == "Dataset Explorer":
    st.subheader("Dataset preview and basic analysis")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df.head(100), use_container_width=True)
    if "isFraud" in df.columns:
        st.write("Label distribution")
        st.bar_chart(df["isFraud"].value_counts().rename(index={0:"Legitimate",1:"Fraud"}))
    st.write("Missing values")
    st.dataframe(df.isna().sum().rename("missing_count"))
    st.download_button("Download current dataset", df.to_csv(index=False).encode("utf-8"),
                       "transactions_dataset.csv", "text/csv")

else:
    st.subheader("About this project")
    st.markdown("""
    **Aim:** Detect potentially fraudulent financial transactions using machine learning.

    **Algorithms:** Logistic Regression and Random Forest.

    **Workflow:** Data collection → preprocessing → feature engineering → model training →
    evaluation → transaction scoring → report generation.

    **Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn and Streamlit.

    **Important limitations:** This is an educational prototype. Demo data is synthetic and does
    not represent real bank activity. Predictions must not be used to block payments or make
    financial decisions without validation, monitoring, security controls and human review.
    """)
    st.markdown("**Expected input columns:** `amount`, `type`, `hour`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `location`, `currency`, and optionally `isFraud`.")
