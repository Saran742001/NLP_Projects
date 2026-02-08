from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    """
    Creates and returns a TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=3000,      # limit vocabulary size
        ngram_range=(1, 1),     # unigrams for now
        min_df=2                # ignore very rare words
    )
    return vectorizer
