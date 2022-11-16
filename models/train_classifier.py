import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pickle


def load_data(database_filepath):
    """
    Function to load data from database
    Args:
    database_filepath
    Return:
    X, Y, category_names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', con=engine)  
    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
    category_names=list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    """
    Function to split text into words and lemmatize them
    Args:
    text: Input text (str)
    Return:
    clean_tokens: list of words
    """
    # find urls and replace with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function to build and improve model
    Args:
    None
    Return:
    cv: improved model with help of GridSearch
    """
    pipeline_ada=Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    # define parameters for AdaBoostClassifier GridSearch
    parameters_ada=parameters = {
        'vect__max_features': (None, 5000),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [0.1, 0.7]
    }
    # perform GridSearch
    cv = GridSearchCV(pipeline_ada, param_grid=parameters_ada)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to test model (prints f1 score, precision and recall)
    Args:
    model: pipeline, X_test, y_test
    Return:
    None
    """
    # make predictions
    y_pred = model.predict(X_test)

    i = 0
    for col in y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    

def save_model(model, model_filepath):
    """
    Function to save model in pickle file
    Args:
    model, model_filepath
    Return:
    None
    """
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)



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