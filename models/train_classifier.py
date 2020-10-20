# import libraries
import pandas as pd
import numpy as np

import pickle
import re
import nltk

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

from scipy.stats import hmean
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    """
    Load Data from database into dataframe
    
    Arguments:
        database_filepath: path to SQLite db
        
    Output:
        X: feature DataFrame
        y: label DataFrame
        category_names: disaster category names list
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    
    category_names = df.columns[4:]
    
    X = df['message']
    y = df.iloc[:,4:]
    
    return X, y

def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Arguments:
    text: string containing message for processing
       
    Returns:
    Tokens: list of strings. 
    List containing normalized and stemmed word tokens
    """

    urls = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_find = re.findall(urls, text)
    for url in urls_find:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens_cleaned = []
    for tok in tokens:
        token_s = [tok for tok in tokens if tok not in stopwords.words('english')]
        token_s = lemmatizer.lemmatize(tok).lower().strip()
        tokens_cleaned.append(token_s)

    return tokens_cleaned

def build_model():
    """
    Build model function
    
    Returns:
        pipeline: sklearn.model_selection.GridSearchCV. 
        It contains a sklearn estimator.
    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
        ))
    ])

    # GridSearch parameters
    parameters = {
        'clf_estimator_lr': [0.1, 0.3],
        'clf_estimator_estimators': [100, 200]
    }

    # GridSearch
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=3)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate model function
    
    Arguments:
        model: sklearn.model_selection.GridSearchCV.  
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    # Predict categories of messages
    y_pred = model.predict(X_test)

    # Print performance metric for each category (accuracy, precision, recall and f1_score)
    for i in range(0, len(category_names)):
        print(category_names[i])
        print("\t Accuracy: {:.4f}\t\t Precision: {:.4}\t\t Recall: {:.4}\t\t F1 Score:{:.4}".format(
            accuracy(y_true[:,i], y_pred[:,i]),
            precision(y_true[:,i], y_pred[:,i], average='weighted'),
            recall(y_true[:,i], y_pred[:,i], average='weighted'),
            f1(y_true[:,i], y_pred[:,i], average='weighted')
        ))


def save_model(model, model_filepath):
    """
    Save model function
    
    Arguments:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Trained model is saved as pickel into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
