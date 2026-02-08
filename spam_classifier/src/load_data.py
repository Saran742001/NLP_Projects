# src/load_data.py
import pandas as pd

def load_dataset(path="data/spam.csv"):
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines='skip')
    df = df[['label','message']]
    df.dropna(inplace=True)
    return df
