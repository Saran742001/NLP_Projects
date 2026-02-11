import pandas as pd

def load_dataset(path="data/news.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
