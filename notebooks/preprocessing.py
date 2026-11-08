# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Diabetes: Hospital Readmission
#
# ### Data preprocessing
#
# **Core Question:** At the moment of hospital discharge, estimate the risk that a diabetic patient will be readmitted as an inpatient in less
# than 30 days, to prioritize additional follow-up.
#
# This notebook performs all data cleaning and preprocessing procedures, based on the decisions in the data visualization notebook. The clean dataset is stored in the `data/clean` folder, and the `data/processed` folder will store the data that has already been preprocessed and split into train/validation/test datasets.
#
# ### Index
#
# 0. Setup
# 1. Loading data
# 2. Data cleaning and preparation
# 3. Train/Validation/Test split
# 4. Learned preprocessing
#

# %%
# allows automatic reloading of the external modules that are imported in this notebook (created in the src/ folder)
# %reload_ext autoreload
# %autoreload 2

# %% [markdown]
# ### 0. Setup
#

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# make src/ importable so that we can import functions from there
PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import (
    read_raw_data,
    read_id_mappings,
    clean_data,
    add_features,
    assign_split,
)
from src.features import make_set, build_preprocessor

BLUE, ORANGE, GREY, TEXT, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#b4b2a9",
    "#0b0b0b",
    "#52514e",
    "#e6e5e0",
)

pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:.3f}".format)
plt.rcParams["figure.dpi"] = 100
print("Project folder:", PROJECT_ROOT)

# %% [markdown]
# ### 1. Loading data
#
# Only `?` is read as missing, so that `None` in `max_glu_serum` and `A1Cresult` stays as a `Not measured` value.
#

# %%
df_raw = read_raw_data()
ids = read_id_mappings()
print(f"Dataset shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

# %% [markdown]
# ### 2. Data preparation
#
# These data preparation steps are fixed rules, explained in the data visualization notebook `notebooks/data_visualization.ipynb`. In short, the preparation steps are:
#
# 1. **Exclude** encounters that ended in death or hospice
# 2. **Target**: `readmitted == "<30"` to 1, otherwise 0
# 3. **Decode IDs and group**: Group admission types/sources and discharge groups. "No information" codes to `Unknown`.
# 4. **Missing categories as a level**: race, payer_code, medical_specialty to `Unknown`; gender `Unknown/Invalid` to `Unknown`
# 5. **Lab results**: `None` to `Not measured`
# 6. **ICD-9 grouping** of the three diagnoses: (`src/dataset.map_idc_9_to_band`)
# 7. **Age band to ordinal**: [0-10], ..., [90-100] to 0..9
# 8. **Binary**: `change` (Ch to 1, otherwise 0), `diabetesMed` (Yes to 1, otherwise 0)
# 9. **Drop**: `weight`, `examide`, `citoglipton`
# 10. **Feature engineering**: Add `n_prior_visits`, `n_med_changes`, `n_diabetes_meds`
#

# %%
df_clean = clean_data(df_raw)
config.CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(config.CLEAN_DATA_DIR / f"clean.csv", index=False)
print("Saved the clean set to", config.CLEAN_DATA_DIR)

df_full = add_features(df_clean)

print(f"Raw data rows: {len(df_raw)}")
print(f"Clean data rows: {len(df_full)}")
print(f"Removed data rows: {len(df_raw) - len(df_full)}")

print(f"\nTarget rate: {df_full[config.TARGET].mean():.3%}")

# Display altered columns
df_full[
    [
        "age",
        "age_ordinal",
        "diag_1",
        "diag_1_group",
        "discharge_disposition_description",
        "discharge_group",
        "max_glu_serum",
        "n_prior_visits",
        "n_diabetes_medications",
        "n_dosage_changes",
        config.TARGET,
    ]
].head(10)

# %%
remaining = df_full.isna().sum()
remaining[remaining > 0]

# %% [markdown]
# As we can see, only the `diag_x` columns have missing values (the "None" strings). This is by design, since the `diag_x_group` columns group these into "Missing".
#

# %% [markdown]
# ### 3. Train / validation / test split
#
# As determined in the data visualization step, the split will be done **by patient**, to ensure that no patient appears simultaneously in the training and validation/test datasets.
#
# Additionally, since the data is naturally unbalanced (only 11,4% <30days readmission rate), we will use a `StratifiedGroupKFold` to ensure that the target rate is kept between training, validation and test sets.
#
# Instead of splitting directly into train, validation and test sets, we will assign a column to identify what rows go into which fold. Then, learned preprocessing will only be assigned to training rows, avoiding data leakage.
#

# %%
df_full["split"] = assign_split(df_full)

overlap = df_full.groupby("patient_nbr")["split"].nunique().gt(1).sum()
print("Patients in more than one set:", overlap)

# %% [markdown]
# ## 4. Learned preprocessing
#
# Anything that is **estimated** from the data is learned from the training rows and then applied unchanged to validation and test.
#
# We will create column arrays based on the type of preprocessing each column needs, and then create a `ColumnTransformer` to handle each group.
#
# **Group summary** (according to the decisions on `notebooks/data_visualization`)
# | Columns | Steps |
# |---|---|
# | skewed numeric (`config.SKEWED`) | clip at the 1st/99th percentile -> Yeo-Johnson (standardised) |
# | other numeric (`config.NUMERIC`) | clip at the 1st/99th percentile -> standardise |
# | age band (`config.ORDINAL`) | standardise only: clipping would merge the youngest bands. Replaces original `age` column |
# | binary (`change`, `diabetesMed`) | unchanged |
# | categories (`config.CATEGORICAL`) | one-hot encoded; categories under 1% of training rows share one "Infrequent" column, along with unseen categories |
#

# %%
train = df_full[df_full["split"] == "train"]
validation = df_full[df_full["split"] == "validation"]
test = df_full[df_full["split"] == "test"]

X_train, y_train = make_set(train)
X_val, y_val = make_set(validation)
X_test, y_test = make_set(test)

preprocessor = build_preprocessor()
preprocessor.fit(X_train, y_train)
Xt_train = preprocessor.transform(X_train)
Xt_val = preprocessor.transform(X_val)
Xt_test = preprocessor.transform(X_test)
print(f"Input columns : {X_train.shape[1]}")
print(f"Output columns: {Xt_train.shape[1]}")

# %% [markdown]
# #### 4.1 Yeo-Johnson: before and after
#
# Yeo-Johnson was the chosen method to ensure it handles 0 in columns such as `number_inpatient`. The previous clipping already removed extreme cases and outliers.
#

# %%
yeo_johnson = preprocessor.named_transformers_["skewed"].named_steps["yeo_johnson"]

fig, axes = plt.subplots(len(config.SKEWED), 2, figsize=(13, 2.6 * len(config.SKEWED)))

for i, col in enumerate(config.SKEWED):
    before, after = X_train[col], Xt_train[col]
    axes[i, 0].hist(before, bins=40, color=BLUE, edgecolor="white")
    axes[i, 0].set_title(f"{col}  |  before, skew = {before.skew():.2f}", fontsize=10)
    axes[i, 1].hist(after, bins=40, color=ORANGE, edgecolor="white")
    axes[i, 1].set_title(f"after, skew = {after.skew():.2f}", fontsize=10)
plt.suptitle("Yeo-Johnson: before vs after", fontsize=14, y=1.0)
plt.tight_layout()
plt.show()

# %%
skew_summary = pd.DataFrame(
    {
        "skew_before": X_train[config.SKEWED].skew(),
        "skew_after": Xt_train[config.SKEWED].skew(),
        "lambda": yeo_johnson.lambdas_,
    }
)
skew_summary["improvement"] = (
    skew_summary["skew_before"].abs() - skew_summary["skew_after"].abs()
)
skew_summary.sort_values("skew_before", ascending=False)

# %% [markdown]
# #### 4.3 Rare categories
#
# As mentioned previously, rare categories (less than 1% representation) are bundled into an "infrequent" category.
#

# %%
encoder = preprocessor.named_transformers_["categorical"]
infrequent = [
    list(infreq) if infreq is not None else []
    for infreq in encoder.infrequent_categories_
]
pd.DataFrame(
    {
        "categories_in_train": [len(c) for c in encoder.categories_],
        "output_columns": [
            len(c) - len(inf) + bool(inf)
            for c, inf in zip(encoder.categories_, infrequent)
        ],
        "n_merged": [len(inf) for inf in infrequent],
        "merged_as_infrequent": infrequent,
    },
    index=config.CATEGORICAL,
)

# %% [markdown]
# With the preprocessing done, we will store the processed data in `data/processed` for feature selection and modelling
#

# %%
config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
for name, X, y in [
    ("train", Xt_train, y_train),
    ("validation", Xt_val, y_val),
    ("test", Xt_test, y_test),
]:
    X.assign(**{config.TARGET: y.values}).to_csv(
        config.PROCESSED_DATA_DIR / f"{name}.csv", index=False
    )
print("Saved the three processed sets to", config.PROCESSED_DATA_DIR)
