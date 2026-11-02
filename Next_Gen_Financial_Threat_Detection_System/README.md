# Next Generation Financial Threat Detection System

A complete academic prototype based on the supplied project synopsis. It uses Logistic Regression and Random Forest to classify financial transactions, estimates fraud probability, supports single and batch screening, displays a dashboard, compares model metrics, and exports CSV reports.

## 1. Project structure

```text
Next_Gen_Financial_Threat_Detection_System/
│ app.py
│ train_model.py
│ generate_dataset.py
│ requirements.txt
│ README.md
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── train.py
├── data/
│   └── demo_transactions.csv
├── models/
│   └── model_bundle.joblib  (created after training)
└── reports/
```

## 2. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for installing Python packages

## 3. Run in VS Code (Windows)

1. Extract the ZIP file.
2. Open VS Code → **File → Open Folder** → select the extracted project folder.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   ```

   If Python 3.10 is not installed, use `python -m venv .venv`.

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this once in that terminal:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   Then activate again.

6. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Train the models:

   ```powershell
   python train_model.py
   ```

8. Start the web application:

   ```powershell
   streamlit run app.py
   ```

9. Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## 4. Alternative: create a fresh demo dataset

```powershell
python generate_dataset.py
python train_model.py
streamlit run app.py
```

## 5. Dataset format

For training, upload a labelled CSV containing `isFraud` (0 = legitimate, 1 = fraud). The following fields are used; missing optional fields are filled with defaults.

| Column | Meaning |
|---|---|
| amount | Transaction amount in original currency |
| currency | INR, USD, EUR, GBP, AED or SGD |
| type | PAYMENT, TRANSFER, CASH_OUT, CASH_IN or DEBIT |
| hour | Transaction hour (0–23) |
| oldbalanceOrg | Sender balance before transaction |
| newbalanceOrig | Sender balance after transaction |
| oldbalanceDest | Recipient balance before transaction |
| newbalanceDest | Recipient balance after transaction |
| location | Usual, New or International |
| isFraud | Target label: 0 or 1 (required for training) |

The application also recognizes `is_fraud` as an alternative target name. Keep the same feature definitions and units in training and inference data. `transaction_id` is retained for reporting but is not used as a model feature.

## 6. Features

- Synthetic labelled demo dataset generator.
- Data cleaning, missing-value handling and feature engineering.
- Currency conversion to INR using fixed demonstration rates in `src/data_utils.py`.
- Logistic Regression and Random Forest comparison.
- Accuracy, precision, recall, F1-score and ROC-AUC.
- Best model saved to `models/model_bundle.joblib`.
- Single transaction fraud probability and risk level.
- Batch CSV screening and downloadable full / flagged reports.
- Dashboard with predicted fraud distribution and amount analysis.

## 7. Important academic and safety notes

- The bundled demo data is **synthetic**. It is created for demonstrating the workflow and does not represent real customers, banks or actual fraud patterns.
- Currency conversion values are fixed illustrative rates, not live exchange rates. Update them from a trusted source for a real deployment.
- A fraud probability is a model score, not proof that a transaction is fraudulent.
- This prototype is not connected to any bank/payment gateway and does not block transactions.
- Real deployment requires lawful data access, privacy/security controls, independent validation on representative data, threshold calibration, drift monitoring, audit logs, human review and operational testing.
- For a real dataset, check class imbalance and avoid evaluating only by accuracy. Recall, precision, F1, ROC-AUC and the confusion matrix are important.

## 8. Troubleshooting

**`streamlit` is not recognized**
```powershell
python -m streamlit run app.py
```

**`ModuleNotFoundError`**
Make sure `.venv` is activated and run:
```powershell
pip install -r requirements.txt
```

**Model not found**
Run `python train_model.py` before opening the dashboard.

**Training CSV error**
Confirm the file is a valid CSV and contains both 0 and 1 in `isFraud`.

## 9. Suggested project demonstration

1. Open Dataset Explorer and show the demo data.
2. Open Train / Evaluate Model and train both algorithms.
3. Explain the comparison table and selected model.
4. Open Dashboard and explain total, predicted fraud and fraud rate.
5. Enter a transaction in Single Transaction Check.
6. Upload a CSV in Batch Fraud Detection and download the report.
