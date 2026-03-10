import os
import sys

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.load_data import load_recipes
from src.preprocessing import clean_text

def run():
    print("👨‍🍳 Cook AI - Entry Point")
    print("-" * 20)
    
    df = load_recipes()
    if df.empty:
        print("Dataset not found. Please add recipes to data/recipes.csv.")
        return

    print(f"Loaded {len(df)} recipes.")
    
    # Test text cleaning
    sample_text = "Cooking 2 plates of Spagetti with Tomato & Basil!"
    print(f"Sample cleaning: {sample_text} -> {clean_text(sample_text)}")
    print("-" * 20)
    print("Type 'streamlit run app/streamlit_app.py' to launch UI.")

if __name__ == "__main__":
    run()
