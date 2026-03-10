import streamlit as st
import pandas as pd
import sys
import os

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import load_recipes
from src.preprocessing import clean_text

st.set_page_config(page_title="Cook AI", layout="wide")

st.title("👨‍🍳 Cook AI: Smart Recipe Assistant")
st.sidebar.header("Options")

def get_data():
    return load_recipes()

df = get_data()

if not df.empty:
    st.write(f"📊 Loaded {len(df)} recipes")
    
    # Robust search interface
    query = st.text_input("Find a recipe by name or ingredients...")
    if query:
        cleaned_query = clean_text(query)
        st.write(f"🔎 Results for: **{query}**")
        
        # Searching across multiple columns
        mask = (
            df['recipe_name'].str.contains(query, case=False, na=False) |
            df['ingredients'].str.contains(query, case=False, na=False) |
            df['instructions'].str.contains(query, case=False, na=False)
        )
        results = df[mask]
        
        if not results.empty:
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No matches found. Try different ingredients!")
else:
    st.info("No recipes found. Please add recipes to `data/recipes.csv`.")

st.sidebar.markdown("---")
st.sidebar.write("Cook AI v1.0")
