from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_tfidf_vectorizer(max_features=5000):
    """Returns a configured TF-IDF Vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9
    )

def get_count_vectorizer(max_features=5000):
    """Returns a configured Count Vectorizer."""
    return CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9
    )
