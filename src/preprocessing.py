import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from src.utils import *

def impute_data(X_train, X_test, imputer=None):
    """
    Performs imputation of missing values on training and test feature matrices.

    By default, uses an IterativeImputer with 10 iterations and a fixed
    random seed for reproducibility. The imputer is fit on the training data
    and applied to both train and test sets.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix with potential missing values.
        X_test (pd.DataFrame): Test feature matrix with potential missing values.
        imputer (sklearn.impute.IterativeImputer, optional): 
            A pre-initialized imputer. If None, a default IterativeImputer is used.

    Returns:
        tuple:
            - pd.DataFrame: Imputed training feature matrix with the same columns as X_train.
            - pd.DataFrame: Imputed test feature matrix with the same columns as X_test.
    """
    if imputer is None:
        imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    return pd.DataFrame(X_train_imputed, columns=X_train.columns), pd.DataFrame(X_test_imputed, columns=X_test.columns)

def scale_data(X_train, X_test, scaler=None):
    """
    Scales training and test feature matrices using a specified scaler.

    By default, applies MinMax scaling (rescaling features to [0, 1]).
    The scaler is fit on the training data and applied to both train
    and test sets to avoid data leakage.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix to fit and transform.
        X_test (pd.DataFrame): Test feature matrix to transform.
        scaler (sklearn.base.TransformerMixin, optional):
            A scikit-learn scaler instance (e.g., StandardScaler, MinMaxScaler).
            If None, a MinMaxScaler is used.

    Returns:
        tuple:
            - pd.DataFrame: Scaled training feature matrix with the same columns as X_train.
            - pd.DataFrame: Scaled test feature matrix with the same columns as X_test.
    """
    if scaler is None:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

def combine_data(preprocessed_df: pd.DataFrame, unstructured_df: pd.DataFrame, on: str = 'Patient_ID', how: str = 'inner') -> pd.DataFrame:
    """
    Combines a preprocessed structured DataFrame with an embedding-based
    unstructured DataFrame.

    Both DataFrames must contain a common key column (default: "Patient_ID").
    If the key column types differ, both are coerced to string before merging
    to prevent dtype mismatches.

    Parameters:
        preprocessed_df (pd.DataFrame): Preprocessed structured dataset containing
            patient-level features.
        unstructured_df (pd.DataFrame): Embedding dataset aligned to the same
            patients via the join key.
        on (str, optional): Column name to join on. Default is "Patient_ID".
        how (str, optional): Type of join to perform (e.g., "inner", "left").
            Default is "inner".

    Returns:
        pd.DataFrame:
            A combined DataFrame containing both structured features and
            embedding features, joined on the specified key column.
    """
    # Align dtypes
    if preprocessed_df[on].dtype != unstructured_df[on].dtype:
        # Coerce both to string to avoid merge errors
        preprocessed_df[on] = preprocessed_df[on].astype(str)
        unstructured_df[on] = unstructured_df[on].astype(str)

    combined_df = pd.merge(preprocessed_df, unstructured_df, on=on, how=how)
    return combined_df

def preprocess_combined_data(combined_df) -> dict:
    """
    Preprocesses a combined structured + embedding DataFrame for model training.

    Steps performed include:
        - Cleaning column names.
        - Separating features (X) from the target column ("HasMHD").
        - Splitting into stratified train and test sets.
        - Imputing missing values in features.
        - Scaling features using MinMax scaling.

    Parameters:
        combined_df (pd.DataFrame): A DataFrame containing both structured
            patient-level features and embedding features. Must include the
            binary target column "HasMHD".

    Returns:
        dict:
            A dictionary containing the following elements:
                - "df": The original combined DataFrame.
                - "X": Full feature matrix (pd.DataFrame).
                - "y": Target labels (pd.Series).
                - "X_train_scaled": Scaled training feature matrix (pd.DataFrame).
                - "X_test_scaled": Scaled test feature matrix (pd.DataFrame).
                - "y_train": Training labels (pd.Series).
                - "y_test": Test labels (pd.Series).
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