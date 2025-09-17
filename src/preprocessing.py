import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from .utils import *

def impute_data(X_train, X_test, imputer=None):
    if imputer is None:
        imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    return pd.DataFrame(X_train_imputed, columns=X_train.columns), pd.DataFrame(X_test_imputed, columns=X_test.columns)

def scale_data(X_train, X_test, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

def combine_data(preprocessed_df: pd.DataFrame, unstructured_df: pd.DataFrame, on = 'Patient_ID', how='inner') -> pd.DataFrame:
    """
    Pipeline for combining preprocessed dataframe with structured data with dataframe of embeddings.

    Both dfs should have a 'Patient_ID' column on which to combine, otherwise need to specify
    """
    combined_df = pd.merge(preprocessed_df, unstructured_df, on = on, how = how)
    
    return combined_df

def preprocess_combined_data(combined_df) -> dict:
    """

    """
    cleaned_df = clean_column_names(combined_df)
    X = cleaned_df.drop(columns = ['HasMHD'])
    y = cleaned_df['HasMHD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_imputed, X_test_imputed = impute_data(X_train, X_test)

    X_train_final, X_test_final = scale_data(X_train_imputed, X_test_imputed)

    preprocessed_combined_data = {
        'df': combined_df,
        'X': X,
        'y': y,
        'X_train_scaled': X_train_final,
        'X_test_scaled': X_test_final,
        'y_train': y_train,
        'y_test': y_test
        }

    return preprocessed_combined_data