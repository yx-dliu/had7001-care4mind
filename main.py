import pandas as pd
import numpy as np
import yaml

from src.pipeline import preprocess_data

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load data
structured_path = config['data']['test_path']
df = pd.read_parquet(structured_path)

# Structured data processing
df = preprocess_data(df)
df = df[config['features']]

# Combine structured and unstructured data

# Hyperparameter tuning

# Stratified k-fold cross-validation

# Model training and evaluation

# Statistical analyses

if __name__ == '__main__':
    # Testing
    print('Structured Data Shape:', df.shape)
    print('First 10 rows of the DataFrame:')
    print(df['Lab_Risk_Proportion'].head(10))
    # print(list(df.columns))
