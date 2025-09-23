from .preprocessing import *
from .feature_engineering import *
from .evaluation import *
from .training import *
from .visualisation import *
from .utils import *

import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def preprocess_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline for preprocessing structured data in preparation for combining with embeddings
    """
    # Structured data processing
    processed_df = df.copy()
    processed_df.dropna(subset=['ICD-9'], inplace=True)

    ## Create binary columns for mental health diagnoses of interest
    mhd_icds = flatten_dict(config['icd9_groups'])
    processed_df['HasMHD'] = processed_df['ICD-9'].apply(lambda x: is_mh(x, mhd_icds))

    ## Binary encode sex
    processed_df['Sex_Bin'] = processed_df['Sex'].apply(encode_sex)

    ## One-hot encode patient status
    processed_df = encode_patient_status(processed_df, drop_cols=['Status_External', 'Status_Duplicate'], prefix='Status', col='PatientStatus_calc', drop=True)

    ## Count chronic conditions per patient
    chronic_df = pd.read_csv(config['reference']['chronic_conditions'])
    chronic_icd_set = get_chronic_codes(chronic_df, 'Code')

    processed_df = diagnose_chronic_conditions(processed_df, chronic_icd_set)

    ## List and count risks
    processed_df = list_risks(processed_df, risk_col='Risks')
    processed_df = count_risks(processed_df, risk_col='List_Risks', new_col_name='Num_Risks')

    ## List and count comorbidities

    physcomorb_icd_set = get_phys_comorb_codes(df_path=config['reference']['physical_comorbidities'], sheets=config['reference']['physical_comorbidities_sheets'], comorb_col='PhysComorb', code_col='ICD-9')

    processed_df['PhysComorb'] = processed_df['ICD-9'].apply(diagnose_physical_comorbidities, physcomorb_sets=physcomorb_icd_set)
    processed_df = make_physcomorb_onehot(processed_df, physcomorb_sets=physcomorb_icd_set)
    processed_df = count_physcomorb(processed_df)

    ## Recalculate age
    processed_df['Age_2015'] = processed_df.apply(calculate_age, axis=1)

    ## Calculate duration of medications
    processed_df = count_longtermmeds(processed_df, col_name='Med_Durations', new_col_name='LongTermMeds_Num')
    processed_df = count_shorttermmeds(processed_df, col_name='Med_Durations', new_col_name='ShortTermMeds_Num')

    ## Calculate abnormal lab measurements
    labs_combined = combine_labs_by_patient(processed_df)
    processed_df = check_lab_values(processed_df, labs_combined, lab_result_functions)
    processed_df['Lab_Risk_Score'] = processed_df.apply(lambda row: summarize_lab_risk(row['labs'], row['Sex']), axis=1)
    processed_df['Lab_Risk_Proportion'] = processed_df.apply(lambda row: proportion_abnormal(row['labs'], row['Sex']), axis=1)

    return processed_df

def combine_preprocessed_data(structured_df: pd.DataFrame, embedding_sizes: list) -> dict:
    """
    Pipeline for applying preprocessing to combined data. 
    
    Returns dict of combined data by embedding size. No resampling, scaling etc. are done.
    """

    embedding_sizes = config['embedding_sizes']

    combined_dfs = {}
    for size in embedding_sizes:
        combined_dfs[size] = combine_data(structured_df, pd.read_parquet(config['data'][size]))

    combined_preprocessed_data = {}
    for k, v in combined_dfs.items():
        combined_preprocessed_data[k] = v

    return combined_preprocessed_data

def load_models(tuned=True):
    """
    Load models from config, does not include LGBMClassifier
    """
    model_dict = load_models_from_config(tuned=tuned)

    return model_dict

def run_stratified_k_fold_cv(model, training_data, skf_n_splits, seed, 
                             label_col = None, id_col = None, shuffle = True, 
                             under_sample = True, impute = True, impute_max_iter = None,
                             scale = True, save_res = False, plot_names = None):
    """
    Runs stratified k-fold cross-validation for all models using the given X and y

    Need to resample, impute, and scale for each fold

    Returns dict of scores for each fold of a given model
    """
    to_drop = []
    skf = StratifiedKFold(n_splits=skf_n_splits, shuffle=shuffle, random_state=seed)

    if label_col is not None:
        to_drop.append(label_col)

    if id_col is not None:
        to_drop.append(id_col)

    X = training_data.drop(to_drop, axis=1)
    y = training_data[label_col]
    model_scores = {}
    
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train_final, X_test_final, y_train_final, y_test_final = process_stratified_k_fold_data(X, y, train_index, test_index, seed, under_sample, impute, impute_max_iter, scale)

        train_eval_final = {
            'X_train_final': X_train_final,
            'X_test_final': X_test_final,
            'y_train_final': y_train_final,
            'y_test_final': y_test_final
        }

        model_scores[f'fold{fold_idx+1}'] = run_and_evaluate_single_fold(
            model, fold_idx, X_train_final, y_train_final, X_test_final, y_test_final, save_res, plot_names)

    return model_scores, train_eval_final

def run_and_evaluate_single_fold(model, fold_idx, X_train_final, y_train_final, X_test_final, y_test_final, save_res=False, plot_names=None) -> dict:
    
    """
    Trains and evaluates a model on a single cross-validation fold.

    Fits the model, generates predictions and probabilities, computes
    accuracy, AUC, recall, precision, and F1 for both train and test sets,
    and optionally saves plots.

    Returns:
        dict: Dictionary of evaluation metrics for train and test sets.
    """
    
    print(f"Fold: {fold_idx+1}")
    model.fit(X_train_final, y_train_final)

    y_pred_train, y_pred_test, y_prob_train, y_prob_test = evaluate_model(
        model, X_train_final, y_train_final, X_test_final, y_test_final,
        save_res, plot_names
        )

    train_acc = accuracy_score(y_train_final, y_pred_train)
    test_acc = accuracy_score(y_test_final, y_pred_test)

    train_auc = roc_auc_score(y_train_final, y_prob_train)
    test_auc = roc_auc_score(y_test_final, y_prob_test)

    train_recall = recall_score(y_train_final, y_pred_train)
    test_recall = recall_score(y_test_final, y_pred_test)

    train_precision = precision_score(y_train_final, y_pred_train)
    test_precision = precision_score(y_test_final, y_pred_test)

    train_f1 = f1_score(y_train_final, y_pred_train)
    test_f1 = f1_score(y_test_final, y_pred_test)

    print(f'Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')
    print(f'Train AUC: {train_auc}, Test AUC: {test_auc}')
    print(f'Train Recall: {train_recall}, Test Recall: {test_recall}')
    print(f'Train Precision: {train_precision}, Test Precision: {test_precision}')

    score_dict = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_f1': train_f1,
        'test_f1': test_f1
        }

    return score_dict
