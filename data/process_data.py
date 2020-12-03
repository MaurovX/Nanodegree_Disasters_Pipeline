'''
    File name: process_data.py
    Author: Maurice J.
    Date created: 12/11/2020
    Date last modified: 02/12/2020
    Python Version: 3.6
'''
# System
import sys
import pandas as pd

# Data Import/Export
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load_data merges both datasets required for NLP modelling 
    joined on index "id"

    :messages_filepath: Path of database containing text features (csv)
    :categories_filepath: Path of database containing categories features (csv)
       
    :return: df pandas DataFrame
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    clean_data Cleans data for modelling. Generates binary response
    feature dataframe for multiOutput classifiers. Cleans empty features,
    removes duplicates.

    :df: Dataframe of joined datasets 
       
    :return: df pandas DataFrame cleaned for modelling.
    
    """
    # Create categories subset
    categories = df["categories"].str.split(pat=";", expand=True)
    row = categories.iloc[[1]]
    category_colnames = [category_name.split("-")[0] for category_name in row.values[0]]
    categories.columns = category_colnames

    # Separate columns and encode binary response
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Clean binary response
    categories.loc[categories["related"] == 2, "related"] = 1

    # Replace categories column in df with new category columns
    df.drop(["categories"], axis=1, inplace=True)
    df = pd.concat([df, categories], join="inner", axis=1)

    # Drop null columns
    df = df.drop(["child_alone"], axis=1)
    # Remove duplicates
    print("Number of duplicates before removal are:{}".format(sum(df.duplicated())))
    df.drop_duplicates(inplace=True)
    print("Number of duplicates after removal are:{}".format(sum(df.duplicated())))
    return df


def save_data(df, database_filename):
    """
    save_data Exports the input dataframe to an SQLite type data storage

    :df: Dataframe of cleaned data
    :database_filename: Database name (string)

    """
    database_filepath = database_filename
    engine = create_engine("sqlite:///" + database_filepath)
    table_name = "T_disaster_df"
    df.to_sql(table_name, engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
