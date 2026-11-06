## Dataset related helpers to simplify notebooks

import pandas as pd

from src.config import RAW_CSV, IDS_CSV, ICD_9_OTHERS, ICD_9_DIABETES, ICD_9_CHAPTERS

def read_raw_data(path=RAW_CSV):
    return pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)

# map the id mappings CSV into structured id tables
def read_id_mappings(path=IDS_CSV):
    id_tables, current_table = {}, None

    for line in path.read_text().splitlines():
        code, description = line.split(",", 1) # only split on the first comma, because descriptions may contain more
        description = description.strip().strip('"')
        if not code.strip(): # empty row, ignore
            continue
        if code.endswith("_id"): # new ID mapping table
            current_table = code
            id_tables[current_table] = {}
        else: # new ID on the current table
            id_tables[current_table][int(code)] = description.strip()

    return id_tables

def map_idc_9_to_band(code):
    if pd.isna(code):
        return "Missing"
    code = str(code).strip().upper()
    
    if code[0] in ICD_9_OTHERS:
        return ICD_9_OTHERS[code[0]]

    value = int(float(code))
    if value == ICD_9_DIABETES:
        return "Diabetes"

    for start, end, disease in ICD_9_CHAPTERS:
        if start <= value <= end:
            return disease

    return "Missing"
