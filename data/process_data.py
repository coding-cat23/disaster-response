import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load data from two csv files and merge into a single dataframe
    INPUT:
    messages_filepath(str): filepath where messages csv is stored
    categories_filepath(str): filepath where categories csv is stored
    OUTPUT:
    pandas dataframe with the two csv datasets joined
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df




def clean_data(df):
    '''
    Function that cleans messages dataset
    INPUT:
    df(pandas dataframe): dataframe containing messages and categories
    OUTPUT:
    returns cleaned version of df as pandas dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)
    def get_column_names(row):
        '''
        The purpose of this function is to extract a list of new column names for categories
        INPUT: a dataframe with one row
        OUTPUT: a list with column names
        '''
        col_names_list=[]
        for i in range(0,36):
            col_name=row[i].str.split('-')[0][0]
            col_names_list.append(col_name)
        return col_names_list

    category_colnames = get_column_names(row)
    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column]=categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    #We will replace the 2's with 1's, assuming the represent a 'true' value
    df['related'] = df['related'].replace([2],1)
    #We also observe that the child alone column only contains 0's. This feature is not helpful as it does not differentiate our observations from each other, and can be removed as it makes no difference and causes an error in some ML classifiers
    df.drop(columns='child_alone', inplace=True)

    return df
  

    

def save_data(df, database_filename):
    '''
    Function that saves cleaned pandas dataframe into a sqlite database
    INPUT:
    df(pandas dataframe): cleaned dataframe containing messages
    database_filename(str): filename for the database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()