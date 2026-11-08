## Paths and settings used to simplify notebooks.

# Unifies variable names, ensuring that notebooks
# remain consistent even when changing values or column names

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(env_file=PROJECT_ROOT / ".env"):
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


# loads the data file names into the environment
load_env_file()

# Data files
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CSV = RAW_DATA_DIR / os.environ.get("RAW_DATA_FILE", "diabetic_data.csv")
IDS_CSV = RAW_DATA_DIR / os.environ.get("IDS_MAPPING_FILE", "IDS_mapping.csv")

# Generated figures and reports
GENERATED = PROJECT_ROOT / "reports" / "generated"

# Random seed for dataset split
SEED = 100

# Data-related constants

# Target feature name
TARGET = "readmitted_30d"

# ID mapping of the discharge disposition after a hospital encounter.
# These refer to IDs that cannot result in readmissions (hospice care and expired)
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# grouping of types of discharge. Used to reduce column cardinality from 30 possible values to 7
DISCHARGE_GROUPS = {
    "Home": [1],
    "Short-term hospital": [2, 10],
    "Care facility": [3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30],
    "Home with care": [6, 8],
    "Left AMA": [7],
    "Outpatient": [9, 12, 16, 17],
    "Unknown": [18, 25, 26],
}

# grouping of types of admission. Joins "no info" into a single category and slots
# trauma center into the emergency group
ADMISSION_TYPE_GROUPS = {
    "Emergency": [1, 7],  # emergency and trauma center
    "Urgent": [2],
    "Elective": [3],
    "Newborn": [4],
    "Unknown": [5, 6, 8],
}

# grouping of sources of admission. Joins "no info" into a single category and reduces cardinality
# from 26 possible values to 6
ADMISSION_SOURCE_GROUPS = {
    "Referral": [1, 2, 3],
    "Transfer": [4, 5, 6, 10, 18, 19, 22, 25, 26],
    "Emergency room": [7],
    "Law enforcement": [8],
    "Unknown": [9, 15, 17, 20, 21],
    "Newborn": [11, 12, 13, 14, 23, 24],
}

# ICD_9 diagnostic codes for supplemental classification or external causes of injury
ICD_9_OTHERS = {"V": "Supplementary factors", "E": "External causes"}

# ICD_9 diagnostic code for diabetes
ICD_9_DIABETES = 250

# ICD_9 diagnostic codes for remaining diagnosis.
# Each value is represented as a tuple of (window_lower_limit, window_upper_limit, classification of disease)
ICD_9_CHAPTERS = [
    (1, 139, "Infectious and parasitic"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine, nutritional and metabolic"),
    (280, 289, "Blood and blood-forming organs"),
    (290, 319, "Mental disorders"),
    (320, 389, "Nervous system and sense organs"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy and childbirth"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital anomalies"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms and ill-defined conditions"),
    (800, 999, "Injury and poisoning"),
]

# List of medication columns
MEDICATIONS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Data variable groups - needed for the ColumnTransformer learned preprocessing

# Threshold for low-representation categorical values
MIN_CATEGORY_THRESHOLD = 0.01

# Medications prescribed in at least 1% of encounters keep their own column
COMMON_MEDICATIONS = [
    "metformin",
    "repaglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
]

# Numeric variables with |skewness| > 1
SKEWED = [
    "time_in_hospital",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "n_prior_visits",
    "n_dosage_changes",
]

# Remaining numeric variables
NUMERIC = ["num_lab_procedures", "number_diagnoses", "n_diabetes_medications"]

# Ordinal codes. Original "age" column is dropped, as it is no longer needed
ORDINAL = ["age_ordinal"]

BINARY = ["change", "diabetesMed"]

CATEGORICAL = [
    "race",
    "gender",
    "admission_type",
    "discharge_group",
    "admission_source",
    "payer_code",
    "medical_specialty",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    *COMMON_MEDICATIONS
]

FINAL_FEATURES = SKEWED + NUMERIC + ORDINAL + BINARY + CATEGORICAL
