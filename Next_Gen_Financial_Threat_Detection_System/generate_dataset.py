from src.data_utils import generate_demo_data

if __name__ == "__main__":
    df = generate_demo_data(10000)
    df.to_csv("data/demo_transactions.csv", index=False)
    print(f"Created {len(df)} synthetic transactions at data/demo_transactions.csv")
