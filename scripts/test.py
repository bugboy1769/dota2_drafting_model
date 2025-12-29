import pandas as pd
data = pd.read_pickle("data/processed/val.pkl")
print(f"Mean Outcome: {sum(x['outcome'] for x in data) / len(data)}")