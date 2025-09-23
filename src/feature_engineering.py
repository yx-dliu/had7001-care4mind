import pandas as pd
from .utils import *

### ICD-9 Code Cleaning Functions

def clean_icd9(row, n):
    """
    Clean a list of ICD-9 diagnosis codes.

    - Drops codes that start with a letter (non-numeric).
    - Converts remaining codes to floats, then formats them with `clean_icd_code`.
    - Returns the cleaned list of codes, or a sentinel value `n` if the row is empty
      after cleaning or if conversion fails.

    Parameters:
        row (list): List of ICD-9 codes (strings or numbers).
        n: Value to return if the row cannot be cleaned (e.g., None, np.nan, or a marker).

    Returns:
        list or n: Cleaned ICD-9 codes as strings, or `n` if problematic.
    """
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
    """
    Normalize a single ICD code string or number.

    Converts the code to float, then back to string while removing 
    unnecessary trailing zeros and decimal points (e.g., '250.00' → '250'). 
    Returns None if the code cannot be converted.

    Parameters:
        code (str or float): ICD code to clean.

    Returns:
        str or None: Cleaned ICD code as string, or None if invalid.
    """
    try:
        # Convert to float in case it's a string like '250.00'
        code_float = float(code)
        # Convert back to string, drop trailing zeros *after* decimal
        code_str = str(code_float).rstrip('0').rstrip('.')
        return code_str
    except ValueError:
        return None  # or handle as needed (e.g., log bad value)
    
def check_overlap(codes, chronic_icd_set):
    """
    Check overlap between a list of codes and a reference set of chronic ICD codes.

    Parameters:
        codes (list): List of ICD codes to check.
        chronic_icd_set (set): Reference set of chronic ICD codes.

    Returns:
        tuple:
            - list: Codes found in both input list and reference set.
            - int: Number of overlapping codes.
    """
    overlap = set(codes) & chronic_icd_set
    return list(overlap), len(overlap)

### REDUNDANT ICD-9 CODE CLEANING FUNCTIONS!!    
def clean_icd_code(code):
    # If the code ends in '.0', drop the decimal; else, keep it
    return str(int(code)) if code == int(code) else str(code)

def is_mh(codes, icds):
    """
    Check whether any mental health ICD codes are present.

    Parameters:
        codes (list): List of patient ICD codes.
        icds (list or set): Reference collection of mental health ICD codes.

    Returns:
        int: 1 if there is at least one overlap, 0 otherwise.
    """
    return int(bool(set(codes) & set(icds)))
    
### Binary Encoding Sex Function

def encode_sex(value):
    """
    Encode sex/gender into a binary variable.

    Converts 'male'/'m' → 0 and 'female'/'f' → 1.
    Returns None for unrecognized, missing, or non-string values.

    Parameters:
        value (str): Input value representing sex/gender.

    Returns:
        int or None: 0 for male, 1 for female, or None if invalid.
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
    """
    Extract and clean ICD codes from a DataFrame column, returning them as a unique set.

    Applies `clean_icd_code` to each value in the specified column and
    collects the results into a set of unique codes.

    Parameters:
        df (pd.DataFrame): DataFrame containing ICD codes.
        code_col (str): Name of the column with raw ICD codes.

    Returns:
        set: Unique set of cleaned ICD codes.
    """
    df['Str_Code'] = df[code_col].apply(clean_icd_code)
    chronic_icd = list(df['Str_Code'])
    chronic_icd_set = set(chronic_icd)
    return chronic_icd_set

def diagnose_chronic_conditions(df: pd.DataFrame, chronic_icd_set: set) -> pd.DataFrame:
    """
    Identify chronic conditions in each row of a DataFrame.

    For each row's list of cleaned ICD codes, checks overlap with the
    provided chronic ICD code set. Adds two new columns:
        - 'Chronic_Diagnoses': List of overlapping chronic codes.
        - 'Num_Chronic': Count of overlapping chronic codes.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Clean_ICD' column with lists of ICD codes.
        chronic_icd_set (set): Set of chronic ICD codes to check against.

    Returns:
        pd.DataFrame: Original DataFrame with added chronic diagnosis columns.
    """
    df['Chronic_Diagnoses'], df['Num_Chronic'] = zip(*df['Clean_ICD'].apply(lambda codes: check_overlap(codes, chronic_icd_set)))
    return df

### Risks Function

def list_risks(df: pd.DataFrame, risk_col: str, new_col_name: str='List_Risks') -> pd.DataFrame:
    """
    Normalise a risk column so all entries are lists.

    Converts missing, NaN, or non-list values in `risk_col` to empty lists
    and stores the result in a new column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        risk_col (str): Name of the column containing risk values.
        new_col_name (str): Name of the output column to store lists (default 'List_Risks').

    Returns:
        pd.DataFrame: DataFrame with an added column of risk values as lists.
    """
    df[new_col_name] = df[risk_col].apply(
    lambda x: [] if x is None or (not isinstance(x, list) and pd.isna(x)) else x
    )
    return df

def count_risks(df: pd.DataFrame, risk_col: str, new_col_name: str='Num_Risks') -> pd.DataFrame:
    """
    Count the number of risks in each row of a DataFrame.

    Assumes the specified column contains lists of risks. Adds a new column
    with the length of each list.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        risk_col (str): Name of the column containing lists of risks.
        new_col_name (str): Name of the output column to store counts (default 'Num_Risks').

    Returns:
        pd.DataFrame: DataFrame with an added column of risk counts.
    """
    df[new_col_name] = df[risk_col].apply(lambda x: len(x))
    return df

### Physical Comorbidities Functions
def get_phys_comorb_codes(df_path: str, sheets: list, comorb_col: str, code_col: str) -> set:
    """
    Extract physical comorbidity codes from Excel sheets.

    Reads one or more sheets from an Excel file, splits comma-separated ICD
    codes into lists, and builds a dictionary mapping comorbidity names to
    sets of associated codes.

    Parameters:
        df_path (str): Path to the Excel file.
        sheets (list): List of sheet names to read.
        comorb_col (str): Column name containing comorbidity names.
        code_col (str): Column name containing raw ICD code strings.

    Returns:
        dict: Mapping of comorbidity name → set of cleaned ICD codes.
    """
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
    """
    Identify physical comorbidities based on a patient's ICD codes.

    Compares the given list of ICD codes against predefined sets of codes
    for each comorbidity and returns the matching comorbidity names.

    Parameters:
        icd_codes (list): List of ICD codes for a patient.
        physcomorb_sets (dict): Mapping of comorbidity name → set of ICD codes.

    Returns:
        list: Names of comorbidities diagnosed for the patient.
    """
    icd_set = set(icd_codes)
    diagnoses = []
    for k, v in physcomorb_sets.items():
        if icd_set & v:
            diagnoses.append(k)
    return diagnoses

def make_physcomorb_onehot(df: pd.DataFrame, physcomorb_sets: dict, suffix: str='_Bin') -> pd.DataFrame:
    """
    Create one-hot encoded columns for physical comorbidities.

    For each comorbidity in `physcomorb_sets`, adds a binary indicator column
    showing whether the patient’s 'PhysComorb' list includes codes for that condition.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'PhysComorb' column with ICD codes.
        physcomorb_sets (dict): Mapping of comorbidity name → set of ICD codes.
        suffix (str): Suffix to append to each new column name (default '_Bin').

    Returns:
        pd.DataFrame: DataFrame with additional binary indicator columns.
    """
    for k, _ in physcomorb_sets.items():
        df[k + suffix] = df['PhysComorb'].apply(lambda x: check_Bin(x, physcomorb_sets[k]))
    return df

def count_physcomorb(df: pd.DataFrame, col_name: str='PhysComorb', new_col_name: str='Num_PhysComorb') -> pd.DataFrame:
    """
    Count the number of physical comorbidities per row.

    Computes the length of each list in the specified column and stores
    the result in a new column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column containing lists of physical comorbidities (default 'PhysComorb').
        new_col_name (str): Name of the output column to store counts (default 'Num_PhysComorb').

    Returns:
        pd.DataFrame: DataFrame with an added column of comorbidity counts.
    """
    df[new_col_name] = df[col_name].apply(len)
    return df

### Function to recalculate age
def calculate_age(row):
    """
    Calculate age based on birth year and (optional) deceased year.

    If 'DeceasedYear' is missing (NaN), age is computed as 2015 minus
    'BirthYear'. Otherwise, it is computed as 'DeceasedYear' minus
    'BirthYear'.

    Parameters:
        row (pd.Series): A row from a DataFrame containing 'BirthYear'
                         and 'DeceasedYear' columns.

    Returns:
        int or float: Calculated age in years.
    """
    if pd.isna(row['DeceasedYear']):
        age = 2015 - row['BirthYear']
    else:
        age = row['DeceasedYear'] - row['BirthYear']
    return age

### Function to recalculate medication duration

def count_longtermmeds(df: pd.DataFrame, col_name: str='Med_Durations', new_col_name: str='LongTermMeds_Num') -> pd.DataFrame:
    """
    Count the number of long-term medications per row.

    Long-term medications are defined as those with duration > 30 days.
    If the value in `col_name` is not a list, assigns -1.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column containing lists of medication durations (default 'Med_Durations').
        new_col_name (str): Name of the output column to store counts (default 'LongTermMeds_Num').

    Returns:
        pd.DataFrame: DataFrame with an added column of long-term medication counts.
    """
    df[new_col_name] = df[col_name].apply(
    lambda durs: sum((d is not None and not pd.isna(d) and d > 30) for d in durs) if isinstance(durs, list) else -1)
    return df

def count_shorttermmeds(df: pd.DataFrame, col_name: str='Med_Durations', new_col_name: str='ShortTermMeds_Num') -> pd.DataFrame:
    """
    Count the number of short-term medications per row.

    Short-term medications are defined as those with duration ≤ 7 days.
    If the value in `col_name` is not a list, assigns -1.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column containing lists of medication durations (default 'Med_Durations').
        new_col_name (str): Name of the output column to store counts (default 'ShortTermMeds_Num').

    Returns:
        pd.DataFrame: DataFrame with an added column of short-term medication counts.
    """
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
    """
    Clean and convert a lab value to float.

    Handles numeric types directly, and for strings removes symbols
    like '>', '<', '=', and '~' before conversion. Returns None if the
    value cannot be parsed as a float.

    Parameters:
        value (int, float, or str): Raw lab value.

    Returns:
        float or None: Cleaned numeric lab value, or None if invalid.
    """
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
    """
    Classify a fasting glucose lab result.

    Uses thresholds to categorize the value:
        - 2: Diabetes (≥ 7.0 mmol/L)
        - 1: Impaired fasting glucose (> 6.1 and < 7.0 mmol/L)
        - 0: Normal (≤ 6.1 mmol/L)
        - None: Missing or unparseable result

    Parameters:
        test (dict): Dictionary-like object with a 'result' field.

    Returns:
        int or None: Category label (0, 1, or 2), or None if invalid.
    """
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
