"""
Utility functions for cleaning ICD codes, checking overlaps, and preprocessing DataFrames.
"""

import re
import pandas as pd

def count_nas(df):
    """
    Summarize missing values per column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Columns:
            - 'col': column name
            - 'nas': number of NaNs
            - 'percent_nas': percentage of NaNs (descending order)
    """
    tot = len(df)
    cols = list(df.columns)
    nas = [df[col].isna().sum() for col in cols]
    percent_nas = [na/tot * 100 for na in nas]

    na_df = pd.DataFrame.from_dict({"col": cols, "nas": nas, "percent_nas": percent_nas})
    na_df.sort_values(by = ["percent_nas"], ascending = False, inplace = True)
    na_df = na_df.reset_index(drop = True)

    return na_df

def flatten_dict(to_flatten: dict) -> list:
    """
    Flattens a nested dictionary into a list of tuples.
    
    Args:
        dict (dict): The dictionary to flatten.
        
    Returns:
        list: list of values in dictionary.
    """
    dict_list = [v for values in to_flatten.values() for v in values]
    return dict_list

def check_overlap(codes, icd_set):
    """
    Find overlapping codes between a list and an ICD set.

    Args:
        codes (list): Codes to check.
        icd_set (set): Reference set of ICD codes.

    Returns:
        tuple: (list of overlaps, count of overlaps)
    """
    overlap = set(codes) & icd_set
    return list(overlap), len(overlap)

def check_bin(codes, icd_set):
    """
    Check if any codes match a given ICD set and return a binary flag.

    Args:
        codes (list or scalar): Codes to check.
        icd_set (set): Reference set of ICD codes.

    Returns:
        int: 1 if overlap exists, 0 otherwise.
    """
    if isinstance(codes, list):
        return int(any(code in icd_set for code in codes))
    if pd.isna(codes):
        return 0

    return int(codes in icd_set)

def clean_column_names(df):
    """
    Clean DataFrame column names by replacing special characters with underscores.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = [re.sub(r'[\"\'\[\]\{\}\\,()]+', '_', col) for col in df.columns]
    return df
