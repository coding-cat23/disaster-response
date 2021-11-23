import sys
import pandas as pd
import re
import os
import pickle
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
import nltk 
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier  
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
        Function to load data from sql database
        INPUT: database_filepath(str) the filepath location of the DATABASE
        OUTPUT: X(pandas dataframe) with message features
                y(pandas dataframe) with message classification labels
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df.message.values #Extract messages
    y = df.iloc[:,4:] #Extract classification results 
    return X,y


def tokenize(text):
    '''
    Function to split message text into tokens and returns root form of the words
    INPUT: message text (string)
    OUTPUT: clean_tokens: list of words (strings) in root form 
    '''
    #normalize text
    normalized_text=text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #convert to tokens
    tokens = word_tokenize(normalized_text)
    
    #remove all stopwords
    words = [t for t in tokens if t not in stopwords.words("english")]
    
    #find root form of words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for w in words:
        clean_tok = lemmatizer.lemmatize(w).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    '''
    Function to build machine learning model that classifies the messages
    OUTPUT: logistic regression model optimised by grid search
    '''
    #Logistic Regression Pipeline
    logreg_pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',  MultiOutputClassifier(LogisticRegression()))
        ])

    parameters = {
    'tfidf__use_idf': (True, False),
    'clf__estimator__random_state': [None,42],
    'clf__estimator__solver':['liblinear','lbfgs']       
}
    #Fit a grid search model
    cv = GridSearchCV(logreg_pipeline, param_grid=parameters)

    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate classification model
    INPUT:  model is a classification ML model
            X_test: test messages
            Y_test: classificaion labels
    OUTPUT: prints precision, recall and f1 score
    '''
    cv_y_pred = model.predict(X_test)
    i=0
    
    for col in Y_test:
        print("Feature: ",col)
        print(classification_report(Y_test[col],cv_y_pred[:, i]))
        i +=1
    accuracy = (cv_y_pred == Y_test.values).mean()
    print("Model Accuracy: ",accuracy)


def save_model(model, model_filepath):
    '''
    Function to save ML model as .pkl file to a filepath
    INPUT:
    model is a classification ML model
    model_filepath(str) the filepath location of where the .pkl file should be saved
    '''
    # Create a pickle file for the model
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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