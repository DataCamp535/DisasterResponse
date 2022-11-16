import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load and merge datasets
    Args:
    messages_filepath: filepath message dataset
    categories_filepath: filepath categories dataset
    Return:
    df: merged dataframe
    """
    # load message data
    messages = pd.read_csv(messages_filepath)
    # load categories data
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories)
    return df

def clean_data(df):
    """
    Function to clean data (split into different columns for each category and set values to 0 and 1, duplicates removed)
    Args:
    df: input dataframe
    Return:
    df: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames= [sub[:-2] for sub in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert to numeric values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # replace values >1 with 1
    for column in categories:
        categories[column]=np.where(categories[column]>1, 1, categories[column])
    # drop 'categories' column in original df
    del df['categories']
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df=df.drop_duplicates(subset=None, keep='first', inplace=False)
    return df


def save_data(df, database_filename):
    """
    Function to save df in database
    Args:
    df: dataframe
    database_filename: filename of database
    Return:
    None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)
    pass  


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