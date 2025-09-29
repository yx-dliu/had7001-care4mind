import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from .preprocessing import *
from .feature_engineering import *
from .evaluation import *
from .training import *
from .utils import *

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def preprocess_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses structured clinical data in preparation for combining with embeddings.

    Steps performed include:
        - Dropping rows with missing ICD-9 codes.
        - Creating a binary indicator for mental health diagnoses of interest.
        - Binary encoding patient sex.
        - One-hot encoding patient status.
        - Counting chronic conditions per patient using a reference file.
        - Listing and counting risk factors.
        - Identifying and encoding physical comorbidities by ICD-9 code.
        - Recalculating patient age relative to 2015.
        - Counting long-term and short-term medications.
        - Aggregating lab measurements by patient and applying lab-based
          risk classification functions.
        - Computing per-patient lab risk scores and proportions.

    Parameters:
        df (pd.DataFrame): The raw structured patient-level DataFrame containing
            clinical variables such as ICD-9 codes, demographics, status, and labs.

    Returns:
        pd.DataFrame:
            A processed DataFrame with additional derived features, including:
                - HasMHD (binary indicator for mental health diagnosis).
                - Sex_Bin (binary-encoded sex).
                - One-hot encoded patient status columns.
                - Chronic condition counts and flags.
                - Physical comorbidity indicators and counts.
                - Recalculated age (Age_2015).
                - LongTermMeds_Num and ShortTermMeds_Num.
                - Lab_Risk_Score and Lab_Risk_Proportion.
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
    Combines preprocessed structured data with embedding features for multiple
    embedding sizes.

    This function reads embedding data (from parquet files specified in the
    configuration) for each embedding size and merges them with the structured
    clinical data. No additional preprocessing, scaling, or resampling is
    applied at this stage.

    Parameters:
        structured_df (pd.DataFrame): Preprocessed structured data containing
            patient-level features.
        embedding_sizes (list): A list of embedding sizes (e.g., ["pca_128",
            "pca_256"]) to combine with the structured data. Values are
            overridden by `config["embedding_sizes"]`.

    Returns:
        dict:
            A dictionary mapping each embedding size (str) to a combined
            DataFrame containing both structured and embedding features.
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
    Loads a set of machine learning models from configuration.

    Parameters:
        tuned (bool, optional): If True, loads models with tuned hyperparameters
            (from the "tuned_models" section of the config). If False, loads
            untuned baseline models. Default is True.

    Returns:
        dict:
            A dictionary mapping model names (str) to initialized scikit-learn
            estimator objects. Note that the LightGBM classifier (LGBMClassifier)
            is not included here and must be loaded separately with
            `load_lgbm`.
    """
    model_dict = load_models_from_config(tuned=tuned)

    return model_dict

def load_lgbm(y_train_final, tuned = True, tuned_params = None, handle_imbalance = True):
    """
    Loads a LightGBM (LGBMClassifier) model with optional tuned hyperparameters
    and handling for class imbalance.

    Parameters:
        y_train_final (array-like): Training labels, used to calculate
            class imbalance ratio if `handle_imbalance` is True.
        tuned (bool, optional): If True, a tuned LightGBM model is expected.
            Requires `tuned_params` to be provided. Default is True.
        tuned_params (dict, optional): Dictionary of hyperparameters for the
            tuned LightGBM model. Must be provided if `tuned=True`.
        handle_imbalance (bool, optional): If True, adjusts `scale_pos_weight`
            based on the ratio of negative to positive classes in
            `y_train_final`. Default is True.

    Returns:
        LGBMClassifier:
            A LightGBM classifier initialized with the specified tuned
            hyperparameters and imbalance handling (if enabled).
    """
    if tuned:
        if tuned_params is None:
            raise ValueError("Must pass tuned hyperparameters")
        elif handle_imbalance:
            counter = Counter(y_train_final)
            scale_pos_weight = counter[0]/counter[1]
            tuned_params['scale_pos_weight'] = scale_pos_weight
            lgb = LGBMClassifier(**tuned_params)
            
    return lgb

def run_stratified_k_fold_cv(model, training_data, skf_n_splits, seed, 
                             label_col = None, id_col = None, shuffle = True, 
                             under_sample = True, impute = True, impute_max_iter = None,
                             scale = True, save_res = False, plot_names = None):
    """
    Runs stratified k-fold cross-validation for a given model and dataset.

    For each fold:
        - Splits the data into train and test sets with stratified sampling.
        - Optionally performs undersampling, imputation, and scaling.
        - Trains the model on the training set and evaluates it on the test set.
        - Optionally saves plots and results for each fold.

    Parameters:
        model: scikit-learn-compatible estimator to be trained and evaluated.
        training_data (pd.DataFrame): DataFrame containing features, labels, and optional ID columns.
        skf_n_splits (int): Number of folds for stratified k-fold CV.
        seed (int): Random seed for reproducibility.
        label_col (str, optional): Name of the column containing labels. Required.
        id_col (str, optional): Name of the column containing unique IDs. If provided, dropped before training.
        shuffle (bool, optional): Whether to shuffle data before splitting. Default is True.
        under_sample (bool, optional): Whether to apply undersampling of majority class. Default is True.
        impute (bool, optional): Whether to perform iterative imputation for missing values. Default is True.
        impute_max_iter (int, optional): Maximum number of iterations for imputation. Used only if `impute=True`.
        scale (bool, optional): Whether to apply scaling to features. Default is True.
        save_res (bool, optional): Whether to save per-fold results to disk. Default is False.
        plot_names (dict, optional): Dictionary mapping plot types (e.g., "confusion_matrix", "roc_curve")
            to filenames for saving. Default is None.

    Returns:
        tuple:
            - model_scores (dict): Mapping from fold name (e.g., "fold1") to evaluation results for that fold.
            - train_eval_final (dict): Dictionary containing the processed train/test splits from the last fold.
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
    Trains and evaluates a model on a single stratified cross-validation fold.

    For the specified fold:
        - Fits the model on the training data.
        - Generates predictions and predicted probabilities for both train and test sets.
        - Computes evaluation metrics (accuracy, AUC, recall, precision, F1).
        - Optionally saves evaluation plots (confusion matrix, ROC curve).

    Parameters:
        model: scikit-learn-compatible estimator to train and evaluate.
        fold_idx (int): Index of the current CV fold (0-based).
        X_train_final (pd.DataFrame or array-like): Feature matrix for the training set.
        y_train_final (pd.Series or array-like): Labels for the training set.
        X_test_final (pd.DataFrame or array-like): Feature matrix for the test set.
        y_test_final (pd.Series or array-like): Labels for the test set.
        save_res (bool, optional): Whether to save evaluation results/plots. Default is False.
        plot_names (dict, optional): Mapping of plot type (e.g., "confusion_matrix", "roc_curve")
            to file paths for saving. Default is None.

    Returns:
        dict:
            Dictionary containing evaluation metrics for both training and test sets:
                - train_acc, test_acc
                - train_auc, test_auc
                - train_recall, test_recall
                - train_precision, test_precision
                - train_f1, test_f1
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
