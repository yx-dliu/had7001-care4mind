from .preprocessing import *
from .feature_engineering import *
from .evaluation import *
from .training import *
from .visualisation import *
from .utils import *

import yaml
import pandas as pd

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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