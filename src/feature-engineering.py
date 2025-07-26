import pandas as pd


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

### Lab Value Functions
# {'FASTING GLUCOSE', 'GLUCOSE TOLERANCE', 
# 'URINE ALBUMIN CREATININE RATIO', 'LDL', 
# 'TRIGLYCERIDES', 'INR', 'GFR', 'MICROALBUMIN', 
# 'HBA1C', 'TOTAL CHOLESTEROL', 'HDL'}

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
    
### Functions to recalculate age
def calculate_age(row):
    if pd.isna(row['DeceasedYear']):
        age = 2015 - row['BirthYear']
    else:
        age = row['DeceasedYear'] - row['BirthYear']
    return age



