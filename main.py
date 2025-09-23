import pandas as pd
import yaml
import json

from src.pipeline import (
    preprocess_structured_data, 
    combine_preprocessed_data,
    load_models,
    run_stratified_k_fold_cv)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load data
structured_path = config['data']['test_path']
df = pd.read_parquet(structured_path)

# Preprocessing structured data
df = preprocess_structured_data(df)
df = df[config['features']]
print(list(df.columns))

# Combine and preprocess structured and unstructured data
combined_preprocessed_data = combine_preprocessed_data(df, config['embedding_sizes']) # dict

# Stratified k-fold cross-validation and evaluation
models_dict = load_models()
results_dict = {}

for embedding_size, embedding_size_data in combined_preprocessed_data.items():
    """
    embedding_size follows the format pca_{embedding size}
    
    embedding_size_data is a Pandas df with the associated training data

    STILL HAVE NOT ADDED CODE FOR RUNNING LGBM
    """
    for model_name, model in models_dict.items():
        print(f"Running stratified k-fold CV for {model_name}")

        plot_names = {
            'confusion_matrix': f'{model_name}_{embedding_size}_confusion_matrix',
            'roc_curve': f'{model_name}_{embedding_size}_roc_curve'
        }
        
        # run_stratified_k_fold_cv optionally returns processed training and test
        # data for each fold
        model_scores, _ = run_stratified_k_fold_cv(
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

        with open(f"{model_name}_{embedding_size}_results.json", "w") as f:
            json.dump(results_dict, f, indent = 4)

# Statistical analyses



if __name__ == '__main__':
    # Testing
    print('Structured Data Shape:', df.shape)
    print('First 10 rows of the DataFrame:')
    print(df['Lab_Risk_Proportion'].head(10))
    # print(list(df.columns))
