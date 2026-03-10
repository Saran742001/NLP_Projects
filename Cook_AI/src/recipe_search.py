from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_recipes(query_vector, recipe_matrix, df, top_n=5):
    similarities = cosine_similarity(query_vector, recipe_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results
