import pandas as pd
import os

def load_recipes(path=None):
    if path is None:
        # Default to data/recipes.csv relative to this project root
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(BASE_DIR, "data", "recipes.csv")
        
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        # Ensure string columns are safe for searching
        for col in ['recipe_name', 'ingredients', 'instructions']:
            if col in df.columns:
                df[col] = df[col].fillna('')
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_recipes()
    print(f"Loaded {len(df)} recipes.")
