## Dataset related helpers to simplify notebooks

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    RAW_CSV,
    IDS_CSV,
    ICD_9_OTHERS,
    ICD_9_DIABETES,
    ICD_9_CHAPTERS,
    TARGET,
    SEED,
)
from src.config import (
    EXCLUDE_DISCHARGE_IDS,
    ADMISSION_SOURCE_GROUPS,
    ADMISSION_TYPE_GROUPS,
    DISCHARGE_GROUPS,
    MEDICATIONS,
)


def read_raw_data(path=RAW_CSV):
    return pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)


# map the id mappings CSV into structured id tables
def read_id_mappings(path=IDS_CSV):
    id_tables, current_table = {}, None

    for line in path.read_text().splitlines():
        code, description = line.split(
            ",", 1
        )  # only split on the first comma, because descriptions may contain more
        description = description.strip().strip('"')
        if not code.strip():  # empty row, ignore
            continue
        if code.endswith("_id"):  # new ID mapping table
            current_table = code
            id_tables[current_table] = {}
        else:  # new ID on the current table
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


def clean_data(data_raw: pd.DataFrame):
    data = data_raw.copy()
    id_mappings = read_id_mappings()

    # Exclude encounters after which a readmission cannot happen.
    data = data[~data["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)]

    # Restructure target: 1 for readmissions in <30 days, 0 otherwise
    data[TARGET] = (data["readmitted"] == "<30").astype(int)

    # Transform categorical IDs (Admission type, Admission Source and Discharge type) into descriptions and group similar IDs
    mappings = [
        ("admission_type_id", ADMISSION_TYPE_GROUPS, "admission_type"),
        ("discharge_disposition_id", DISCHARGE_GROUPS, "discharge_group"),
        ("admission_source_id", ADMISSION_SOURCE_GROUPS, "admission_source"),
    ]
    for column, groups, name in mappings:
        description_column = column.replace(
            "_id", "_description"
        )  # turn, for example, admission_type_id column name into admission_type_description

        data[description_column] = data[column].map(id_mappings[column])

        code_mapping = _invert_groups_codes(
            groups
        )  # transform "Description: [1,2]" into {1: "Description", 2: "Description"}
        data[name] = (
            data[column].map(code_mapping).fillna("Unknown")
        )  # map and join missing data in the Unknown category

    # Group missing categorical data into the Unknown category
    data["race"] = data["race"].fillna("Unknown")
    data["gender"] = data["gender"].replace("Unknown/Invalid", "Unknown")
    data["payer_code"] = data["payer_code"].fillna("Unknown")
    data["medical_specialty"] = data["medical_specialty"].fillna("Unknown")

    # Replace "None" (NaN value) with "Not measured" for glucose and A1C exams
    for column in ["max_glu_serum", "A1Cresult"]:
        data[column] = data[column].replace("None", "Not measured")

    # Map diagnosis to their ICD-9 groups
    for column in ["diag_1", "diag_2", "diag_3"]:
        data[f"{column}_group"] = data[column].map(map_idc_9_to_band)

    # Convert age bands to ordinal values (0..9)
    age_bands = []
    for age in range(0, 100, 10):
        age_band = f"[{age}-{age + 10})"
        age_bands.append(age_band)

    data["age_ordinal"] = data["age"].map({band: i for i, band in enumerate(age_bands)})

    # Transform Yes/No columns to binary values
    data["change"] = (data["change"] == "Ch").astype(int)
    data["diabetesMed"] = (data["diabetesMed"] == "Yes").astype(int)

    # Drop not needed columns
    data = data.drop(
        columns=["weight", "examide", "citoglipton"]
    )  # weight had 96% missing values, others are medications that are never prescribed (constant)
    return data.reset_index(drop=True)


def add_features(clean_data: pd.DataFrame):
    full_data = clean_data.copy()

    # Add n_prior_visits (sum of 3 visit-related column values)
    full_data["n_prior_visits"] = full_data[
        ["number_outpatient", "number_emergency", "number_inpatient"]
    ].sum(axis=1)

    # get all medication columns and sum values (n_dia)
    medications = full_data[[m for m in MEDICATIONS if m in full_data.columns]]
    full_data["n_diabetes_medications"] = (medications != "No").sum(axis=1)
    full_data["n_dosage_changes"] = medications.isin(["Up", "Down"]).sum(axis=1)

    return full_data


def assign_split(data, seed=SEED):
    # shuffle data before creating folds
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    split = np.full(
        len(data), "train", dtype=object
    )  # create a column to store the split assigned to the rows. Init with "train" for all
    for fold, (_, index) in enumerate(
        folds.split(data, data[TARGET], data["patient_nbr"])
    ):
        if fold == 0:  # first of the 5 folds (20% of data) is the test fold
            split[index] = "test"
        elif fold == 1:  # second of the 5 folds (20% of data) is the validation fold
            split[index] = "validation"

    # all other folds (60% of data) are training folds
    return pd.Series(split, index=data.index, name="split")


def _invert_groups_codes(groups: dict):
    result = {}

    for name, codes in groups.items():
        for code in codes:
            result[code] = name

    return result
