import pandas as pd

def load_dataset(path):
    df = pd.read_csv(
        path,
        encoding="latin-1",
        sep=",",
        quoting=3,
        on_bad_lines="skip"
    )
    return df

