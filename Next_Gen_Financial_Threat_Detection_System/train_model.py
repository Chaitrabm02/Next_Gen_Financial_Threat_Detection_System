import pandas as pd
from src.data_utils import generate_demo_data
from src.train import train_models

if __name__ == "__main__":
    path = "data/demo_transactions.csv"
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        data = generate_demo_data(5000)
        data.to_csv(path, index=False)
    print("Training Logistic Regression and Random Forest...")
    print(pd.DataFrame(train_models(data)).T)
    print("Saved model to models/model_bundle.joblib")
