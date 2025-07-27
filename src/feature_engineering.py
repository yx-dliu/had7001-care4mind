import pandas as pd
import yaml
from .utils import *

### ICD-9 Code Cleaning Functions

def clean_icd9(row, n):
    # Drop codes that start with a letter
    row = [code for code in row if not str(code)[0].isalpha()]
    
    if len(row) == 0:
        return n  # signal this row as problematic
    else:
        try:
            num_row = [float(code) for code in row]
            str_row = [clean_icd_code(code) for code in num_row]
            return str_row
        except:
            return n

def clean_icd_code(code):
    try:
        # Convert to float in case it's a string like '250.00'
        code_float = float(code)
        # Convert back to string, drop trailing zeros *after* decimal
        code_str = str(code_float).rstrip('0').rstrip('.')
        return code_str
    except ValueError:
        return None  # or handle as needed (e.g., log bad value)
    
def check_overlap(codes, chronic_icd_set):
    overlap = set(codes) & chronic_icd_set
    return list(overlap), len(overlap)

### REDUNDANT ICD-9 CODE CLEANING FUNCTIONS!!    
def clean_icd_code(code):
    # If the code ends in '.0', drop the decimal; else, keep it
    return str(int(code)) if code == int(code) else str(code)

def is_mh(codes, icds):
    return int(bool(set(codes) & set(icds)))
    
### Binary Encoding Sex Function

def encode_sex(value):
    """
    Converts sex to binary: male = 0, female = 1.
    Returns None for unrecognized values.
    """
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['male', 'm']:
            return 0
        elif value in ['female', 'f']:
            return 1
    return None  # handle unknowns, missing, or unrecognized values

### One-Hot Encoding Patient Status Function

def encode_patient_status(df: pd.DataFrame, drop_cols: list, prefix: str, col: str='PatientStatus_calc', drop: bool=False) -> pd.DataFrame:
    """
    One-hot encodes the PatientStatus column in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the PatientStatus column.
        col (str): The name of the column to encode. Default is 'PatientStatus_calc'.
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns for PatientStatus.
    """
    df[col] = df[col].fillna('Unknown')
    df = pd.get_dummies(df, columns=[col], prefix=prefix, prefix_sep='_')
    df = df.replace({True: 1, False: 0})

    if drop:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

### Chronic Conditions Diagnosis Function

def get_chronic_codes(df: pd.DataFrame, code_col: str) -> set:
    df['Str_Code'] = df[code_col].apply(clean_icd_code)
    chronic_icd = list(df['Str_Code'])
    chronic_icd_set = set(chronic_icd)
    return chronic_icd_set

def diagnose_chronic_conditions(df: pd.DataFrame, chronic_icd_set: set) -> pd.DataFrame:
    df['Chronic_Diagnoses'], df['Num_Chronic'] = zip(*df['Clean_ICD'].apply(lambda codes: check_overlap(codes, chronic_icd_set)))
    return df

### Risks Function

def list_risks(df: pd.DataFrame, risk_col: str, new_col_name: str='List_Risks') -> pd.DataFrame:
    df[new_col_name] = df[risk_col].apply(
    lambda x: [] if x is None or (not isinstance(x, list) and pd.isna(x)) else x
    )
    return df

def count_risks(df: pd.DataFrame, risk_col: str, new_col_name: str='Num_Risks') -> pd.DataFrame:
    df[new_col_name] = df[risk_col].apply(lambda x: len(x))
    return df

### Physical Comorbidities Functions
def get_phys_comorb_codes(df_path: str, sheets: list, comorb_col: str, code_col: str) -> set:

    codes_dict = {}

    for sheet in sheets:
        df = pd.read_excel(df_path, sheet_name=sheet)
        df['Str_Code'] = df[code_col].apply(lambda x: [item.strip() for item in str(x).split(",")])
        codes_dict[sheet] = df['Str_Code'].tolist()

    df['Str_Code'] = df[code_col].apply(lambda x: [item.strip() for item in str(x).split(",")])

    physcomorb_dict = {}

    for pair in zip(df[comorb_col], df['Str_Code']):
        physcomorb_dict[pair[0]] = pair[1]

    physcomorb_sets = {k: set(v) for k, v in physcomorb_dict.items()}

    return physcomorb_sets

def diagnose_physical_comorbidities(icd_codes: list, physcomorb_sets: dict) -> list:
    icd_set = set(icd_codes)
    diagnoses = []
    for k, v in physcomorb_sets.items():
        if icd_set & v:
            diagnoses.append(k)
    return diagnoses

def make_physcomorb_onehot(df: pd.DataFrame, physcomorb_sets: dict, suffix: str='_Bin') -> pd.DataFrame:
    for k, _ in physcomorb_sets.items():
        df[k + suffix] = df['PhysComorb'].apply(lambda x: check_Bin(x, physcomorb_sets[k]))
    return df

def count_physcomorb(df: pd.DataFrame, col_name: str='PhysComorb', new_col_name: str='Num_PhysComorb') -> pd.DataFrame:
    df[new_col_name] = df[col_name].apply(len)
    return df

### Function to recalculate age
def calculate_age(row):
    if pd.isna(row['DeceasedYear']):
        age = 2015 - row['BirthYear']
    else:
        age = row['DeceasedYear'] - row['BirthYear']
    return age

### Function to recalculate medication duration

def count_longtermmeds(df: pd.DataFrame, col_name: str='Med_Durations', new_col_name: str='LongTermMeds_Num') -> pd.DataFrame:
    df[new_col_name] = df[col_name].apply(
    lambda durs: sum((d is not None and not pd.isna(d) and d > 30) for d in durs) if isinstance(durs, list) else -1)
    return df

def count_shorttermmeds(df: pd.DataFrame, col_name: str='Med_Durations', new_col_name: str='ShortTermMeds_Num') -> pd.DataFrame:
    df[new_col_name] = df[col_name].apply(
    lambda durs: sum((d is not None and not pd.isna(d) and d <= 7) for d in durs) if isinstance(durs, list) else -1)
    return df

### Lab Value Functions
# {'FASTING GLUCOSE', 'GLUCOSE TOLERANCE', 
# 'URINE ALBUMIN CREATININE RATIO', 'LDL', 
# 'TRIGLYCERIDES', 'INR', 'GFR', 'MICROALBUMIN', 
# 'HBA1C', 'TOTAL CHOLESTEROL', 'HDL'}

def combine_labs_by_patient(df, id_col='Patient_ID', sex_col='Sex',
                             tests_col='LabTests', dates_col='Lab_Performed_Dates',
                             results_col='Lab_Test_Results', units_col='Lab_UnitOfMeasure'):
    """
    Combines lab test data into a structured format per patient.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing lab-related columns.
        id_col (str): Column name for patient ID.
        sex_col (str): Column name for patient sex.
        tests_col (str): Column name for lab test names.
        dates_col (str): Column name for lab test dates.
        results_col (str): Column name for lab results.
        units_col (str): Column name for units of measure.

    Returns:
        dict: Nested dictionary mapping patient ID to {'sex': ..., 'labs': [...]}
    """
    labs_combined = {}

    for _, row in df.iterrows():
        patient_id = row[id_col]
        sex = row[sex_col]

        if (
            isinstance(row[tests_col], list) and
            isinstance(row[dates_col], list) and
            isinstance(row[results_col], list) and
            isinstance(row[units_col], list)
        ):
            labs = [
                {'test': t, 'date': d, 'result': r, 'units': u}
                for t, d, r, u in zip(
                    row[tests_col],
                    row[dates_col],
                    row[results_col],
                    row[units_col]
                )
            ]
        else:
            labs = []

        labs_combined[patient_id] = {
            'sex': sex,
            'labs': labs
        }

    return labs_combined

def clean_lab_value(value): # some lab values have characters, not just num
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        value = value.strip()
        for symbol in ['>', '<', '=', '~']:
            value = value.replace(symbol, '')
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None

# All functions to ID abnormal values of lab tests based on Canadian thresholds
## In general: 0 = normal, 1 = bordering problematic, 2 = problematic

def check_fasting_glucose(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result >= 7.0:
        return 2
    elif result > 6.1:
        return 1
    else:
        return 0

def check_egfr(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result >= 90:
        return 0
    elif result >= 60:
        return 1
    else:
        return 2

def check_glucose_tolerance(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result >= 11.1:
        return 2
    elif result >= 7.8:
        return 1
    else:
        return 0

def check_hba1c(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result >= 6.5:
        return 2
    elif result >= 6.0:
        return 1
    else:
        return 0

def check_hdl(test, sex):
    result = clean_lab_value(test['result'])
    if result is None or not isinstance(sex, str):
        return None
    if sex.lower() == 'male':
        return 0 if result >= 1.0 else 2
    elif sex.lower() == 'female':
        return 0 if result >= 1.3 else 2
    else:
        return None

def check_inr(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result <= 1.1:
        return 0
    elif 2.0 <= result <= 3.0:
        return 1
    else:
        return 2

def check_ldl(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result < 2.0:
        return 0
    elif result < 3.5:
        return 1
    else:
        return 2

def check_microalbumin(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result < 30:
        return 0
    elif result <= 300:
        return 1
    else:
        return 2

def check_total_cholesterol(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result < 5.2:
        return 0
    elif result < 6.2:
        return 1
    else:
        return 2

def check_triglycerides(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result < 1.7:
        return 0
    elif result < 2.3:
        return 1
    else:
        return 2

def check_uacr(test):
    result = clean_lab_value(test['result'])
    if result is None:
        return None
    if result < 3.0:
        return 0
    elif result <= 30.0:
        return 1
    else:
        return 2

lab_result_functions = {
    'FASTING GLUCOSE': check_fasting_glucose,
    'GFR': check_egfr,
    'GLUCOSE TOLERANCE': check_glucose_tolerance,
    'HBA1C': check_hba1c,
    'HDL': check_hdl,
    'INR': check_inr,
    'LDL': check_ldl,
    'MICROALBUMIN': check_microalbumin,
    'TOTAL CHOLESTEROL': check_total_cholesterol,
    'TRIGLYCERIDES': check_triglycerides,
    'URINE ALBUMIN CREATININE RATIO': check_uacr
    }

def check_lab_values(df: pd.DataFrame, labs_combined: dict, lab_result_functions: dict) -> pd.DataFrame:
    for pid, data in labs_combined.items():
        sex = data['sex']
        for lab in data['labs']:
            test_name = lab['test'].strip().upper()  # Normalize the name
            checker = lab_result_functions.get(test_name)

            if checker:
                if test_name == 'HDL':
                    lab['status'] = checker(lab, sex)  # HDL needs sex
                else:
                    lab['status'] = checker(lab)  # Others do not
            else:
                lab['status'] = None
    
    labs_combined_df = pd.DataFrame.from_dict(labs_combined, orient='index')
    labs_combined_df.index.name = 'Patient_ID'
    labs_combined_df.reset_index(inplace=True)
    labs_combined_df = labs_combined_df.drop(['sex'], axis=1)

    df = df.merge(labs_combined_df, on='Patient_ID', how='left')

    return df

### Functions to quantify risk
def summarize_lab_risk(labs, sex):
    """
    Produces raw count of all instances of abnormal 
    lab values for given patient
    """
    risk_score = 0
    for lab in labs:
        test_name = lab['test'].strip().upper()
        check_fn = lab_result_functions.get(test_name)

        if check_fn:
            try:
                if test_name == 'HDL':
                    risk = check_fn(lab, sex)
                else:
                    risk = check_fn(lab)
                risk_score += 0 if risk is None else risk
            except:
                continue  # skip malformed or failed entries
    return risk_score

def proportion_abnormal(labs, sex):
    """
    Calculates the proportion of abnormal lab values.
    Returns None if no labs were available or evaluable.
    """
    total = 0
    flagged = 0

    for lab in labs:
        test_name = lab.get('test', '').strip().upper()
        check_fn = lab_result_functions.get(test_name)

        if check_fn:
            try:
                risk = check_fn(lab, sex) if test_name == 'HDL' else check_fn(lab)
                if risk is not None:
                    total += 1
                    if risk >= 1:
                        flagged += 1
            except:
                continue

    if total == 0:
        return None  # No evaluable labs
    else:
        return flagged / total
