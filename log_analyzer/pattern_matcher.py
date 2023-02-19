import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
import pickle

def create_pipeline(log_file=None):
    # Create a TfidfVectorizer and a TruncatedSVD transformer to reduce the dimensionality of the data
    vectorizer = TfidfVectorizer(max_features=8, min_df=1, stop_words='english')

    # Set the max_features and n_components based on the number of log messages in the file
    if log_file is not None:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            messages = [line for line in f]
        n_messages = len(messages)
        if n_messages < vectorizer.max_features:
            max_features = n_messages
        else:
            max_features = vectorizer.max_features
        n_components = min(max_features, 10)
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=1, stop_words='english')
    else:
        n_components = min(vectorizer.max_features, 10)

    svd = TruncatedSVD(n_components=n_components)

    # Create a pipeline that applies the vectorizer, SVD, and IsolationForest model
    pipeline = make_pipeline(vectorizer, svd, IsolationForest(contamination=0.01))

    return pipeline

def train_pipeline(pipeline, messages):
    # Check for an empty list of messages
    if not messages:
        print('Error: Empty list of messages')
        return None

    # Fit the pipeline on the log messages
    try:
        pipeline.fit(messages)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print('Error: No terms remain after pruning. Try a lower min_df or a higher max_df')
        elif "contain stop words" in str(e):
            print('Error: Only stop words are found in the document. Try lowering the max_df')
        else:
            print('Error:', e)
        return None

    return pipeline

def predict_anomalies(pipeline, messages):
    anomaly_scores = pipeline.decision_function(messages)
    is_anomaly = anomaly_scores < 0

    return is_anomaly
