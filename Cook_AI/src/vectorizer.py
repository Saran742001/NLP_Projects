from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class RecipeVectorizer:
    def __init__(self, max_features=3000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, path="models/tfidf_vectorizer.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        
    def load(self, path="models/tfidf_vectorizer.pkl"):
        self.vectorizer = joblib.load(path)
