import yaml
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.impute import IterativeImputer

from lightgbm import LGBMClassifier

from src.utils import clean_column_names
from collections import Counter

with open('src/grids.yaml', 'r') as file:
    grids = yaml.safe_load(file)

with open('src/models.yaml', 'r') as file:
    models_config = yaml.safe_load(file)

CLASS_MAP = {
    'KNeighborsClassifier': KNeighborsClassifier,
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'LGBMClassifier': LGBMClassifier
}

def tune_model(model, X_train, y_train, grid):
    """
    Conducts hyperparameter tuning for a given model using RandomizedSearchCV.

    Parameters:
        model: A scikit-learn compatible estimator to be tuned.
        X_train (pd.DataFrame or array-like): Training feature matrix.
        y_train (pd.Series or array-like): Training labels.
        grid (dict): Dictionary of hyperparameter search spaces to sample from.

    Returns:
        dict:
            The best set of hyperparameters (as key-value pairs) identified
            during randomized search, based on ROC AUC scoring.
    """

    search = RandomizedSearchCV(model, param_distributions = grid, n_iter = 6, 
                                scoring = 'roc_auc', cv = 5, random_state = 42)
    
    search.fit(X_train, y_train)

    best_params = search.best_params_

    return best_params

def load_models_from_config(models_config = models_config, tuned = True) -> dict:
    """
    Loads machine learning models from a configuration dictionary.

    The configuration must specify model classes and hyperparameters under
    "tuned_models" and "untuned_models". Models are initialized with either
    tuned or untuned parameters depending on the `tuned` flag.

    Note:
        This function does not currently handle weight calculation for
        LightGBM classifiers (LGBMClassifier).

    Parameters:
        models_config (dict): Dictionary containing model specifications.
            Expected structure:
                {
                    "tuned_models": {
                        model_name: {
                            "class": str,     # model class name
                            "params": dict    # tuned hyperparameters
                        },
                        ...
                    },
                    "untuned_models": {
                        model_name: {
                            "class": str,     # model class name
                            "params": dict or "None"
                        },
                        ...
                    }
                }
        tuned (bool, optional): If True, loads models with tuned hyperparameters
            from "tuned_models". If False, loads models with baseline parameters
            from "untuned_models". Default is True.

    Returns:
        dict:
            A dictionary mapping model names (str) to instantiated scikit-learn
            compatible estimator objects.
    """
    model_dict = {}

    if tuned:
        for name, specs in models_config['tuned_models'].items():
            model_class = CLASS_MAP[specs['class']]
            model_dict[name] = model_class(**specs['params'])
    else:
        for name, specs in models_config['untuned_models'].items():
            model_class = CLASS_MAP[specs['class']]
            if specs['params'] == 'None':
                model_dict[name] = model_class()
            else:
                model_dict[name] = model_class(**specs['params'])

    return model_dict

def process_stratified_k_fold_data(X, y, train_index, test_index, seed, under_sample, impute, impute_max_iter, scale):
    """
    Processes training and test splits within stratified k-fold cross-validation.

    For a given fold, this function:
        - Splits the data into training and test sets using provided indices.
        - Optionally applies random undersampling to balance class distribution.
        - Optionally imputes missing values using IterativeImputer.
        - Optionally scales features using StandardScaler.
        - Cleans column names after transformations.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        train_index (array-like): Indices for the training set.
        test_index (array-like): Indices for the test set.
        seed (int): Random seed for reproducibility.
        under_sample (bool): Whether to apply random undersampling to the
            training set.
        impute (bool): Whether to perform imputation of missing values.
        impute_max_iter (int): Maximum number of iterations for iterative
            imputation. Must be provided if `impute=True`.
        scale (bool): Whether to scale features using StandardScaler.

    Returns:
        tuple:
            - X_train_final (pd.DataFrame): Final processed training feature matrix.
            - X_test_final (pd.DataFrame): Final processed test feature matrix.
            - y_train_final (pd.Series): Final training labels (after undersampling).
            - y_test_final (pd.Series): Final test labels (unaltered).
    """
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if under_sample:
        rus = RandomUnderSampler(random_state=seed)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    if impute:
        if impute_max_iter is None:
            raise ValueError("To impute, must set max_iter")
        imputer = IterativeImputer(max_iter=impute_max_iter, random_state=seed)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_resampled), 
                                       columns=X_train_resampled.columns)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), 
                                      columns=X_test.columns)
    else:
        X_train_imputed, X_test_imputed = X_train_resampled, X_test

    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed),
                                      columns=X_train_imputed.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed),
                                     columns=X_test_imputed.columns)
    else:
        X_train_scaled, X_test_scaled = X_train_imputed, X_test_imputed

    X_train_final = clean_column_names(X_train_scaled)
    X_test_final = clean_column_names(X_test_scaled)
    y_train_final = y_train_resampled
    y_test_final = y_test

    return X_train_final, X_test_final, y_train_final, y_test_final
