import pandas as pd
import re

def count_nas(df):

  tot = len(df)
  cols = list(df.columns)
  nas = [df[col].isna().sum() for col in cols]
  percent_nas = [na/tot * 100 for na in nas]

  na_df = pd.DataFrame.from_dict({"col": cols, "nas": nas, "percent_nas": percent_nas})
  na_df.sort_values(by = ["percent_nas"], ascending = False, inplace = True)
  na_df = na_df.reset_index(drop = True)

  return na_df

def flatten_dict(dict: dict) -> list:
    """
    Flattens a nested dictionary into a list of tuples.
    
    Args:
        dict (dict): The dictionary to flatten.
        
    Returns:
        list: list of values in dictionary.
    """
    dict_list = [v for values in dict.values() for v in values]
    return dict_list

def check_overlap(codes, icd_set):
    overlap = set(codes) & icd_set
    return list(overlap), len(overlap)

def check_Bin(codes, icd_set):
    if isinstance(codes, list):
        return int(any(code in icd_set for code in codes))
    elif pd.isna(codes):
        return 0
    else:
        return int(codes in icd_set)
    
def clean_column_names(df):
    df.columns = [re.sub(r'[\"\'\[\]\{\}\\,()]+', '_', col) for col in df.columns]
    return df