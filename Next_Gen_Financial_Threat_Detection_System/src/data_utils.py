import numpy as np
import pandas as pd

CURRENCY_TO_INR = {
    "INR": 1.0, "USD": 83.0, "EUR": 90.0, "GBP": 105.0, "AED": 22.6, "SGD": 61.5
}

def convert_to_inr(amount, currency):
    rate = CURRENCY_TO_INR.get(str(currency).upper(), 1.0)
    return float(amount) * rate

def generate_demo_data(n=5000, random_state=42):
    """Create synthetic, labelled transactions for demonstration only."""
    rng = np.random.default_rng(random_state)
    types = np.array(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
    txn_type = rng.choice(types, n, p=[.35,.18,.25,.12,.10])
    amount = np.clip(rng.lognormal(mean=7.4, sigma=1.15, size=n), 10, 500000)
    hour = rng.integers(0, 24, n)
    old_org = np.clip(rng.lognormal(8.5, 1.1, n), 0, 2_000_000)
    old_dest = np.clip(rng.lognormal(8.0, 1.2, n), 0, 2_000_000)
    new_org = np.maximum(0, old_org - amount + rng.normal(0, 150, n))
    new_dest = old_dest + amount + rng.normal(0, 150, n)
    location = rng.choice(["Usual","New","International"], n, p=[.82,.14,.04])
    currency = rng.choice(list(CURRENCY_TO_INR), n, p=[.72,.10,.05,.04,.05,.04])
    # Synthetic signal: high amount, unusual location/time, transfer/cash-out.
    logit = (-5.0 + 1.15*(amount > 100000) + 1.15*(location != "Usual") +
             .65*((hour <= 4) | (hour >= 23)) + .9*np.isin(txn_type, ["TRANSFER","CASH_OUT"]) +
             .45*(amount > old_org*0.8))
    prob = 1/(1+np.exp(-logit))
    fraud = rng.binomial(1, prob)
    return pd.DataFrame({
        "transaction_id": [f"TXN{i+1:06d}" for i in range(n)],
        "amount": amount.round(2), "currency": currency, "type": txn_type,
        "hour": hour, "oldbalanceOrg": old_org.round(2),
        "newbalanceOrig": new_org.round(2), "oldbalanceDest": old_dest.round(2),
        "newbalanceDest": new_dest.round(2), "location": location, "isFraud": fraud
    })

def prepare_dataframe(df):
    df = df.copy()
    # Accept common alternative target names.
    if "isFraud" not in df.columns and "is_fraud" in df.columns:
        df["isFraud"] = df["is_fraud"]
    defaults = {
        "amount": 0, "currency": "INR", "type": "PAYMENT", "hour": 12,
        "oldbalanceOrg": 0, "newbalanceOrig": 0, "oldbalanceDest": 0,
        "newbalanceDest": 0, "location": "Usual"
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    for col in ["amount","hour","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["currency","type","location"]:
        df[col] = df[col].fillna(defaults[col]).astype(str).str.upper() if col != "location" else df[col].fillna("Usual").astype(str)
    df["amount_inr"] = df.apply(lambda r: convert_to_inr(r["amount"], r["currency"]), axis=1)
    df["balance_delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    return df
