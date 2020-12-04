'''
    File name: train_classifier.py
    Author: Maurice J.
    Date created: 12/11/2020
    Date last modified: 02/12/2020
    Python Version: 3.6
'''
# System
import sys
import os

# Generic
import numpy as np
import pandas as pd

# Data export/import
from sqlalchemy import create_engine
import pickle

# NLP preprocessing
import re
import string

import nltk
from nltk.stem import PorterStemmer
nltk.download('brown')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download('names')
nltk.download('universal_tagset')

# Independent libraries
from normalise import normalise
from normalise import tokenize_basic
import spacy

# Model builders
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filename):
    """
    load_data finds the input for modelling, searches on a SQLite Database
    under the table_name provided in process_data.py.

    :database_filename: Path of database saved by process_data.py
                        eg. # '../data/disaster_response_db.db'
    
    :return:  X dataframe of features (Text) 
              y dataframe of targets. (Binary Responses)
    """
    database_filepath = database_filename  # '../data/disaster_response_db.db'
    engine = create_engine("sqlite:///" + database_filepath)
    table_name = "T_disaster_df"
    df = pd.read_sql_table(table_name, engine)

    # Extract X and y variables from the data for modelling
    X = df["message"]
    y = df.iloc[:, 4:]

    return X, y


def tokenize(text):
    """
    tokenize intents to preprocess the text features, tokenizes and
    lemmatizes each message.

    :text: Dataframe of text features
    
    :return:  clean_tokens as a DataFrame that contains each message
              preprocessed for modelling
    """
    # Convert to lowercase
    text_lw = text.lower()

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text_lw)

    # Lemmanitizer to get the 'grammatical root'  of each word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens standarized
    tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    # Remove stop words
    nltk_stop_words = nltk.corpus.stopwords.words('english')
    tokens_ns = [t for t in tokens if t not in nltk_stop_words]
    
    # Remove punctuation
    clean_tokens = [w for w in tokens_ns if w not in string.punctuation]
   
    return clean_tokens


def build_model():
    """
    build_model defines the pipeline for modelling the text data.
    
    :features: Joins standard preprocessing with StartingVerbExtractor features
    :text_pipeline: Applies tokenize function to text data
    :TfidfTransformer: Vectorizes text
    :starting_verb_transformer: Creates new features under the class StartingVerbExtractor
    :classifier: Defines the model under the wrapper MultiOutputClassifier
    
    :return: Pipeline of steps required for building the NLP model

    """
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    (
                                        "count_vectorizer",
                                        CountVectorizer(tokenizer=tokenize),
                                    ),
                                    ("tfidf_transformer", TfidfTransformer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    evaluate_model prints metrics for the fitted model.
    
    :model: model result of build_model.fit method
    :X_test: DataFrame of test values (Unseen messages)
    :y_test: Target label of unseen texts (True Labels)
    
    :Output: Classification metric reports for each target category

    """
    y_pred = model.predict(X_test)
    print(
        classification_report(y_test.values, y_pred, target_names=y_test.columns.values)
    )


def save_model(pipeline, pickle_filepath):
    """
    save_model saves pipeline output as a picke file.
    
    :pipeline: model result of build_model.fit method
    :pickle_filepath: filepath desired location for pickle file

    """
    pickle.dump(pipeline, open(pickle_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        print(y.shape)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
