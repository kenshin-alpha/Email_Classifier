#Methods related to converting text in into numeric representation and then returning numeric representation may go here

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config

def get_tfidf_embd(df: pd.DataFrame):
    combined_text = df[Config.TICKET_SUMMARY] + " " + df[Config.INTERACTION_CONTENT]
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(combined_text)
    return X