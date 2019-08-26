"""
This module is for your data cleaning.
It should be repeatable.

## PRECLEANING
There should be a separate script recording how you transformed the json api ca
lls into a dataframe and csv.

## SUPPORT FUNCTIONS
cleans part of the dataset and return the partially cleaned data.
"""
import pandas as pd


def remove_first_column(citation):
    """takes dataset as inputs and drop the first column"""
    citation = citation.drop("Unnamed: 0", axis=1)

    return citation


def remove_nulls(citation):
    """takes dataset as inputs and removes missing values from the four columns
"""
    citation = citation[pd.notnull(citation['color'])]
    citation = citation[pd.notnull(citation['driver_city'])]
    citation = citation[pd.notnull(citation['make'])]
    citation = citation[pd.notnull(citation['model'])]

    return citation


def remove_duplicates(citation):
    """takes dataset as input and drops all unique traffic stop ids' from the d
ataset"""
    citation.drop_duplicates(subset='seq_id', keep=False, inplace=True)

    return citation


def full_clean():
    """
    This function runs all the above support functions.

    :return: cleaned dataset to be passed to hypothesis testing and visualizati
on modules.
    """
    dirty_data = pd.read_csv("./data/dirty_data.csv")

    cleaning_data1 = remove_first_column(dirty_data)
    cleaning_data2 = remove_nulls(cleaning_data1)
    cleaned_data = remove_duplicates(cleaning_data2)
    cleaned_data['ticket'] = cleaned_data['violation_type'].apply(lambda x: 1 if x == 'Citation' else 0)

    cleaned_data.to_csv('./data/cleaned_for_testing.csv')

    return cleaned_data
