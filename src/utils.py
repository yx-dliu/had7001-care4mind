import pandas as pd

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