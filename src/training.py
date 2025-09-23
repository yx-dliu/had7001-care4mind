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

def collect_grids(grid_names: list) -> dict:
    RCV_grids = {}
    for name in grid_names:
        RCV_grids[name] = grids[name]
    return RCV_grids

def tune_model(model, X_train, y_train, grid):
    """
    General function for conducting hyperparameter tuning on models using predefined grids and RandomizedSearchCV
    """
    search = RandomizedSearchCV(model, param_distributions = grid, n_iter = 6, 
                                scoring = 'roc_auc', cv = 5, random_state = 42)
    
    search.fit(X_train, y_train)

    best_params = search.best_params_

    return best_params

def make_model_grid():
    return None

def load_models_from_config(models_config = models_config, tuned = True) -> dict:
    # Doesn't load and calculate weights for LGBClassifier yet
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

def load_lgbm(y_train_final, tuned = True, tuned_params = None, handle_imbalance = True):

    if tuned:
        if tuned_params is None:
            raise ValueError("Must pass tuned hyperparameters")
        elif handle_imbalance:
            counter = Counter(y_train_final)
            scale_pos_weight = counter[0]/counter[1]
            tuned_params['scale_pos_weight'] = scale_pos_weight
            lgb = LGBMClassifier(**tuned_params)
    return lgb

def process_stratified_k_fold_data(X, y, train_index, test_index, seed, under_sample, impute, impute_max_iter, scale):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if under_sample:
        rus = RandomUnderSampler(random_state=seed)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
        
    if impute:
        if impute_max_iter is None:
            raise ValueError("To impute, must set max_iter")
        else:
            imputer = IterativeImputer(max_iter=impute_max_iter, random_state=seed)
            X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_resampled), 
                                           columns=X_train_resampled.columns)
            X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)
        
    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), 
                                      columns=X_train_imputed.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed),
                                     columns=X_test_imputed.columns)
            
    X_train_final = clean_column_names(X_train_scaled)
    X_test_final = clean_column_names(X_test_scaled)
    y_train_final = y_train_resampled
    y_test_final = y_test
    
    return X_train_final, X_test_final, y_train_final, y_test_final

