"""
Main script for running stratified k-fold CV on structured + embedding data.
"""

import json

import pandas as pd
import yaml

from src.pipeline import (
    preprocess_structured_data,
    combine_preprocessed_data,
    load_models,
    load_lgbm,
    run_stratified_k_fold_cv
    )

def main():
    """
    Run preprocessing, model training, and evaluation, returning a dict of results.
    """

    with open('config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open('src/models.yaml', 'r', encoding="utf-8") as file:
        models = yaml.safe_load(file)

    # Load data
    structured_path = config['data']['test_path']
    print(f"Loading structured data from {structured_path}")
    df = pd.read_parquet(structured_path)

    # Preprocessing structured data
    df = preprocess_structured_data(df)
    df = df[config['features']]
    df = df.loc[:,~df.columns.duplicated()].copy()
    print(list(df.columns))

    # Combine and preprocess structured and unstructured data
    combined_preprocessed_data = combine_preprocessed_data(df, config['embedding_sizes']) # dict

    # Stratified k-fold cross-validation and evaluation
    models_dict = load_models()
    results_dict = {}

    for embedding_size, embedding_size_data in combined_preprocessed_data.items():

        # embedding_size follows the format pca_{embedding size}
        # embedding_size_data is a Pandas df with the associated training data

        for model_name, model in models_dict.items():
            print(f"Running stratified k-fold CV for {model_name}")

            plot_names = {
                'confusion_matrix': f'{model_name}_{embedding_size}_confusion_matrix',
                'roc_curve': f'{model_name}_{embedding_size}_roc_curve'
            }

            # run_stratified_k_fold_cv optionally returns processed training and test
            # data for each fold
            model_scores, train_eval_final = run_stratified_k_fold_cv(
                model = model,
                training_data = embedding_size_data,
                skf_n_splits = config['skf_n_splits'],
                seed = config['seed'],
                label_col = config['label_col_name'],
                id_col = config['id_col_name'],
                impute_max_iter = config['impute_max_iter'],
                plot_names = plot_names
            )

            results_dict[model_name + "_" + embedding_size] = model_scores

            with open(f"{model_name}_{embedding_size}_results.json", "w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent = 4)
        
        LGB = load_lgbm(
            y_train_final = train_eval_final['y_train_final'],
            tuned = True,
            tuned_params = models['tuned_models']['LGB']['params'],
            handle_imbalance = True
        )

        model_scores, train_eval_final = run_stratified_k_fold_cv(
            model = LGB,
            training_data = embedding_size_data,
            skf_n_splits = config['skf_n_splits'],
            seed = config['seed'],
            label_col = config['label_col_name'],
            id_col = config['id_col_name'],
            impute_max_iter = config['impute_max_iter'],
            plot_names = plot_names
        )
        results_dict['LGB' + "_" + embedding_size] = model_scores
        
        with open(f"LGB_{embedding_size}_results.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent = 4)

    return results_dict


if __name__ == '__main__':
    results = main()
    print("Finished pipeline. Models and datasets run:", list(results.keys()))
