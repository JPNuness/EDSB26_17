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
# ### Data visualization
#
# **Core Question:** At the moment of hospital discharge, estimate the risk that a diabetic patient will be readmitted as an inpatient in less
# than 30 days, to prioritize additional follow-up.
#
# This notebook explores and creates relevant visualizations of the **raw data**, leveraging information to make data cleaning and pre-processing decisions.
#
# Cleaning and pre-processing is done in `notebooks/preprocessing.py`.
#
# **Index**
#
# 0. Setup
# 1. Loading data correctly
# 2. The target variable
# 3. Patients and encounters
# 4. Missing values
# 5. Where did the patients go after the stay
# 6. Admission type
# 7. Demographics
# 8. Diagnoses
# 9. Lab tests
# 10. Medications
# 11. Prior visits
# 12. High-cardinality features: Medical specialty and payer code
# 13. Numeric features and outliers
# 14. Correlation between numeric features
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
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# make src/ importable so that we can import functions from there
PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import read_raw_data, read_id_mappings, map_idc_9_to_band

FIGURES = config.GENERATED / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# colour variables
BLUE, ORANGE, GREY, TEXT, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#b4b2a9",
    "#0b0b0b",
    "#52514e",
    "#e6e5e0",
)

# parametrize matplotlib plots
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
    }
)
# limit float values to 3 decimal places
pd.set_option("display.float_format", "{:.3f}".format)


# helper functions (will be used in a lot of places throughout the notebook, might as well simplify the code)
# saves the figure to the figures folder (reports/generated/figures)
def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")


# limits a string to a given width (50 by default). If shortened, ends with '...'
def shorten(text, width=50):
    if len(text) <= width:
        return text
    else:
        return text[: width - 2].rsplit(" ", 1)[0] + "..."


# Plots the percentage of 1s of the target variable (see section 2 of the notebook) based on the values
# of a given column of the dataframe. Also compares with the overall rate of readmissions.
def rate_bars(
    axes,
    data: str | pd.Series,
    by,
    highlight=(),
    min_n=0,
    horizontal=False,
    sort=False,
    title=None,
):
    """Plot the <30-day readmission rate of each group, against the overall rate.

    Args:
        axes: The matplotlib axes to draw on.
        data (DataFrame): Rows with a 0/1 `target` column.
        by (str or Series): The column (or a Series) that defines the groups.
        highlight (iterable): Groups drawn in the highlight colour.
        min_n (int): Groups with fewer rows are left out (their rate is noise).
        horizontal (bool): Horizontal bars, for long labels.
        sort (bool): Sort the groups by rate instead of by their own order.
        title (str): The chart title.

    Returns:
        DataFrame: The rate and the size of each group shown.
    """
    rates = data.groupby(by, observed=True)["target"].agg(rate="mean", n="size")
    rates = rates[rates["n"] >= min_n]
    if sort:
        rates = rates.sort_values("rate")
    labels = rates.index.astype(str)
    colors = [
        ORANGE if label in set(map(str, highlight)) else BLUE for label in rates.index
    ]
    overall = data["target"].mean()
    if horizontal:
        axes.barh(labels, rates["rate"], color=colors, height=0.7)
        axes.axvline(overall, color=TEXT, linestyle="--", linewidth=1)
        axes.annotate(
            f"all: {overall:.1%}",
            (overall, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=8,
            color=TEXT,
        )
        axes.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes.grid(axis="x")
        axes.grid(axis="y", visible=False)
    else:
        axes.bar(labels, rates["rate"], color=colors, width=0.7)
        axes.axhline(overall, color=TEXT, linestyle="--", linewidth=1)
        axes.annotate(
            f"all: {overall:.1%}",
            (0, overall),
            xycoords=("axes fraction", "data"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color=TEXT,
        )
        axes.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes.set_title(title or f"30-day readmission rate by {by}")
    return rates


print("Project folder:", config.PROJECT_ROOT)
print("Figures folder :", FIGURES)

# %% [markdown]
# ### 1. Loading data correctly
#
# The project guide clarifies that "missingness is field-specific" and that the missing values on the raw CSV are represented by `?`. However, the columns `max_glu_serum` and `A1Cresult` write a "None" string if the tests were not done. This string is interpreted by Pandas as missing.
#
# With that in mind, during preprocessing, we will interpret `?` as the **only** missing value marker. Other "unknown" representations will be treated on a case by case basis.
#

# %% [markdown]
# ### 2. The target variable
#
# The business question is centered around patient readmission in the following 30 days. However, the outcome field (`readmitted`) contains three possible values:
#
# - `<30`
# - `>30`
# - `NO`
#
# In our business case, only `<30` results are interpreted as a binary 1. The other two are interpreted as 0 (not readmitted in the following 30 days).
#

# %%
df = read_raw_data()
id_mappings = read_id_mappings()
df["target"] = (df["readmitted"] == "<30").astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), gridspec_kw={"width_ratios": [3, 2]})
counts = df["readmitted"].value_counts().reindex(["NO", ">30", "<30"])
axes[0].bar(counts.index, counts, color=[GREY, GREY, ORANGE], width=0.6)
axes[0].bar_label(
    axes[0].containers[0], labels=[f"{v / len(df):.1%}" for v in counts], padding=3
)
axes[0].set_title("Source outcome variable")
axes[0].set_ylabel("encounters")

binary = df["target"].value_counts().reindex([0, 1])
axes[1].bar(["0: not <30", "1: <30"], binary, color=[GREY, ORANGE], width=0.6)
axes[1].bar_label(
    axes[1].containers[0], labels=[f"{v / len(df):.1%}" for v in binary], padding=3
)
axes[1].set_title("Target for the project")
plt.tight_layout()
save_figure(fig, "02_target")
plt.show()

# %% [markdown]
# ### 3. Patients and encounters
#
# the `encounter_id` variable is unique, but `patient_nbr` repeats.
#

# %%
print("Nº encounters: ", df.groupby("encounter_id").size().shape[0])
print(
    "Nº encounters for different patients: ", df.groupby("patient_nbr").size().shape[0]
)

# %%
per_patient = df.groupby("patient_nbr").size()
df["n_encounters_per_patient"] = df["patient_nbr"].map(per_patient)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
encounters_distribuition = per_patient.clip(upper=10).value_counts().sort_index()
axes[0].bar(
    [str(i) if i < 10 else "10+" for i in encounters_distribuition.index],
    encounters_distribuition,
    color=[GREY] + [ORANGE] * (len(encounters_distribuition) - 1),
    width=0.7,
)
axes[0].set_title("Patients by number of encounters in the file")
axes[0].set_xlabel("encounters per patient")
axes[0].set_ylabel("patients")

group_encounters = np.where(
    df["n_encounters_per_patient"] > 1,
    "patient with 2+ encounters",
    "patient with 1 encounter",
)
rate_bars(
    axes[1],
    df,
    group_encounters,
    highlight=["patient with 2+ encounters"],
    title="<30-day readmission rate, single vs repeat patients",
)
plt.tight_layout()
save_figure(fig, "03_patient_encounters")
plt.show()

repeat = per_patient > 1
print(
    f"Patients: {per_patient.size:,} | with 2+ encounters: {repeat.sum():,} ({repeat.mean():.1%})"
)
print(
    f"Encounters belonging to previously encountered patients: {df['n_encounters_per_patient'].gt(1).mean():.1%}"
)

# %% [markdown]
# As seen in the charts, About a quarter of the patients appear more than once (23,5%), and they account for close to
# half of all encounters (46,2%). Moreover, about 20% of their encounters are followed by a <30-day readmission, against about 3% for
# patients seen only once.
#
# This makes sense, since a readmission always creates a new encounter in the file. In other words, a patient seen only once is a patient who was not readmitted. However,"number of encounters of this patient" must **never** be a feature, because it introduces target leakage (if different encounters for the same patient would be found in both the training and test sets, the model would be partly scored on previously seen data).
#
# Knowing this, the data will be **grouped by patient**, instead of used as is, ensuring that all encounters for the same patient go into the same dataset (either train, validation or test).
#

# %% [markdown]
# ### 4. Missing values
#

# %%
missing = (df.isna().mean() * 100).sort_values()
missing = missing[missing > 0]

fig, axes = plt.subplots(figsize=(10, 3.8))
colors = [ORANGE if v > 90 else BLUE for v in missing]
axes.barh(missing.index, missing, color=colors, height=0.7)
axes.axvline(20, color=MUTED, linestyle="--", linewidth=1)
axes.axvline(40, color=TEXT, linestyle="--", linewidth=1)
axes.text(20.5, len(missing) - 0.6, "20%: treat", color=MUTED, fontsize=9)
axes.text(40.5, len(missing) - 0.6, "40%: drop", color=TEXT, fontsize=9)
for i, value in enumerate(missing):
    axes.text(value + 0.8, i, f"{value:.1f}%", va="center", fontsize=9, color=MUTED)
axes.set_xlim(0, 105)
axes.set_title("Share of patient encounters with '?' per column")
axes.set_xlabel("missing (%)")
axes.grid(axis="x")
axes.grid(axis="y", visible=False)
save_figure(fig, "04_missing")
plt.show()

# %% [markdown]
# `weight` is almost empty and can be dropped. `medical_specialty` and `payer_code` sit around the 40%
# mark, but they are **categorical** values: we can keep "not recorded" as a category of its
# own instead of assigning a value.
#

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
for ax, col in zip(axes, ["race", "payer_code", "medical_specialty"]):
    is_missing = np.where(df[col].isna(), "missing", "recorded")
    rates = rate_bars(ax, df, is_missing, highlight=["missing"], title=col)
    ax.bar_label(
        ax.containers[0],
        labels=[f"{r:.1%}\nn={int(n):,}" for r, n in rates.values],
        padding=3,
        fontsize=9,
    )
    ax.set_ylim(0, 0.16)
fig.suptitle(
    "<30-day readmission rate when the value is missing vs recorded",
    x=0.01,
    ha="left",
    fontweight="bold",
)
plt.tight_layout()
save_figure(fig, "04_missing_vs_rate")
plt.show()

# %% [markdown]
# As shown in the plots above, `payer_code` and `medical_specialty` present no relevant difference in the target variable when they are present vs. when they are missing. However, encounters with no `race` recorded have a lower readmission rate (2,9% lower), so it makes sense to keep an `Unknown` group for this variable.
#
# In conclusion, in the data preparation step, we will be drop the `weight` column, as it is almost 100% missing. Then, we will keep the other 3 categorical variables with missing data (`payer_code`, `medical_specialty` and `race`) and assign the missing values to a new `Unknown` category (due to the difference in readmission rate in the `race` column when comparing missing and recorded values).
#
# Finally, the missing values in the `diag_` columns will represent a new `Missing` group.
#

# %% [markdown]
# ### 5. Where did the patients go after the stay
#
# Some destinations make readmission impossible. For example, discharges to hospice care or eventual death (13, 14, 19, 20, 21) are terminal care cases. This means that the patients normally won't be readmitted but, exceptionally, these cases are not supposed to be identified as low-risk.
#

# %%
discharge = df.assign(
    discharge=df["discharge_disposition_id"]
    .map(id_mappings["discharge_disposition_id"])
    .map(shorten)
)
excluded_names = set(
    discharge.loc[
        discharge["discharge_disposition_id"].isin(config.EXCLUDE_DISCHARGE_IDS),
        "discharge",
    ]
)

fig, axes = plt.subplots(figsize=(11, 6.5))
rates = rate_bars(
    axes,
    discharge,
    "discharge",
    highlight=excluded_names,
    min_n=100,
    horizontal=True,
    sort=True,
    title="<30-day readmission rate by discharge disposition (groups with 100+ encounters)",
)

for i, n in enumerate(rates["n"]):
    axes.text(
        rates["rate"].iloc[i] + 0.003,
        i,
        f"n={n:,}",
        va="center",
        fontsize=8,
        color=MUTED,
    )
axes.text(
    0.99,
    0.02,
    "orange: hospice care",
    transform=axes.transAxes,
    ha="right",
    color=ORANGE,
    fontsize=9,
)
save_figure(fig, "05_discharge")
plt.show()

excluded_groups = df["discharge_disposition_id"].isin(config.EXCLUDE_DISCHARGE_IDS)
print(f"Expired or hospice care: {excluded_groups.sum():,} encounters")

# %% [markdown]
# To avoid "tricking" the model, we decided to remove the 2423 encounters that ended in hospice care or death (groups with ID in `config.EXCLUDE_DISCHARGE_IDS`).
#

# %%
df["discharge_disposition_id"].value_counts()

# %% [markdown]
# Aditionally, according to the project guide, the `discharge_disposition_id` column has 30 possible values. To reduce cardinality, given that most discharge codes are rare (< 100 cases), we will also group them into discharge groups as follows:
#
# ```python
# DISCHARGE_GROUPS = {
#     "Home": [1],
#     "Short-term hospital": [2, 10],
#     "Care facility": [3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30],
#     "Home with care": [6, 8],
#     "Left AMA": [7],
#     "Outpatient": [9, 12, 16, 17],
#     "Unknown": [18, 25, 26],
# }
# ```
#
# Again, for code reusability, this grouping is a constant in `src/config`.
#

# %%
not_excluded_discharges = df[~excluded_groups]
discharge_groups = {}

for name, codes in config.DISCHARGE_GROUPS.items():
    for code in codes:
        discharge_groups[code] = name

fig, axes = plt.subplots(figsize=(10, 3.8))
rates = rate_bars(
    axes,
    not_excluded_discharges,
    not_excluded_discharges["discharge_disposition_id"]
    .map(discharge_groups)
    .fillna("Unknown")
    .rename("group"),
    horizontal=True,
    sort=True,
    title="<30-day readmission rate by discharge group",
)
for i, n in enumerate(rates["n"]):
    axes.text(
        rates["rate"].iloc[i] + 0.003,
        i,
        f"n={n:,}",
        va="center",
        fontsize=8,
        color=MUTED,
    )
save_figure(fig, "05_discharge_groups")
plt.show()

# %% [markdown]
# ### 6. Admission type
#
# Similarly to the previous section, the `admission_type_id` and `admission_source_id` are categorical variables with large cardinality. We will be looking to reduce this cardinality by:
#
# - Joining "no info" columns: **NULL**, **Not Available**, **Not Mapped**
# - Grouping "related" IDs as follows:
#
# ```python
# ADMISSION_TYPE_GROUPS = {
#     "Emergency": [1, 7],  # emergency and trauma center
#     "Urgent": [2],
#     "Elective": [3],
#     "Newborn": [4],
#     "Unknown": [5, 6, 8],
# }
#
# ADMISSION_SOURCE_GROUPS = {
#     "Referral": [1, 2, 3],
#     "Transfer": [4, 5, 6, 10, 18, 19, 22, 25, 26],
#     "Emergency room": [7],
#     "Law enforcement": [8],
#     "Unknown": [9, 15, 17, 20, 21],
#     "Newborn": [11, 12, 13, 14, 23, 24]
# }
# ```
#

# %%
print(df["admission_type_id"].value_counts())
print(df["admission_source_id"].value_counts())

# %% [markdown]
# ### 7. Demographics
#
# Age is given in 10-year bands, from [0-10] into [90-100].These bands will be coded into ordinal code, 0-9. Regarding gender, we will map the `Unknown/Invalid` values into `Unknown`, for coherence with the rest of the columns.
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(17, 4), gridspec_kw={"width_ratios": [3, 2]})

ages = rate_bars(
    axes[0], not_excluded_discharges, "age", title="Readmission rate by age band"
)
axes[0].set_xticks(
    range(len(ages)), [f"{band}\nn={n:,}" for band, n in ages["n"].items()], fontsize=8
)
axes[0].set_xlabel("")

rate_bars(
    axes[1],
    not_excluded_discharges,
    not_excluded_discharges["race"].fillna("?"),
    highlight=["?"],
    sort=True,
    title="Readmission rate by race",
)
plt.tight_layout()
save_figure(fig, "07_demographics")
plt.show()

# %% [markdown]
# ### 8. Diagnoses
#
# In the dataset, there are three diagnose related columns (`diag_1`, `diag_2` and `diag_3`), 1 for primary diagnosis and 2/3 for secondary diagnosis.
#
# Each of these are encoded according to ICD_9 codes, which range from 1 to 999.
#

# %%
print(
    "Distinct raw codes:",
    {c: not_excluded_discharges[c].nunique() for c in ["diag_1", "diag_2", "diag_3"]},
)

# %% [markdown]
# Despite the codes being numeric, the column is categorical: each code represents a different diagnostic, not a higher or lower value. Since the values are way to many to one-hot encode, we will group the diagnosis columns into bands, according to https://en.wikipedia.org/wiki/List_of_ICD-9_codes .
#
# The encoding will be done according to `src.dataset.map_idc_9_to_band`.
#
# An exception are:
#
# - **Code 250.xx**: diabetes diagnosis. Will be singled out, because the project guide says that "Each encounter includes a diabetes diagnosis".
#
# - **Codes V/E (E001, V01, ...)**: supplemental classification (V) or external causes (E) of injury diagnosis. Will become their own group.
# - **Missing values**: Aggregate into a "Missing" group.
#

# %%
fig, axes = plt.subplots(figsize=(11, 6.5))
icd9_groups = not_excluded_discharges["diag_1"].map(map_idc_9_to_band)
rate_bars(
    axes,
    not_excluded_discharges,
    icd9_groups,
    highlight=["Diabetes"],
    horizontal=True,
    sort=True,
    title="Readmission rate by primary diagnosis ICD-9 group",
)
plt.tight_layout()
save_figure(fig, "08_diagnoses")
plt.show()

# %% [markdown]
# ### 9. Lab tests
#
# Possible values for the lab test columns (`max_glu_serum` and `A1Cresult`) are substance intervals, `Norm` for normal values and `None` for not measured. We will transform `None` values into `Not measured`, one hot encoding the resulting columns.
#

# %%
print(df["A1Cresult"].value_counts())
print(df["max_glu_serum"].value_counts())

# %% [markdown]
# ### 10. Medication columns
#
# Medications given to patients are split into columns, one for each prescription. Possible values include:
#
# - **No**: Did not prescribe that specific medication.
# - **Steady**: Maintained the medication dosage.
# - **Up**: Increased the medication dosage.
# - **Down**: Decreased the medication dosage.
#
# The medication column list is in `src.config.MEDICATIONS`.
#

# %%
meds = not_excluded_discharges[config.MEDICATIONS]
prescribed = (meds != "No").mean().sort_values()

bar_colors = [
    ORANGE if value == 0 else BLUE if value > 0.01 else GREY for value in prescribed
]

fig, axes = plt.subplots(figsize=(15, 6.5))
axes.barh(prescribed.index, prescribed, color=bar_colors, height=0.7)
for i, (drug, value) in enumerate(prescribed.items()):
    if value == 0:
        axes.text(2.2e-6, i, "never prescribed", va="center", fontsize=9, color=TEXT)

axes.set_xscale("log")
axes.set_xlim(2e-6, 1)
axes.axvline(0.01, color=TEXT, linestyle="--", linewidth=1)
axes.text(0.0105, 0.3, "1%", color=TEXT, fontsize=9)
axes.set_ylim(-0.7, len(prescribed) - 0.3)
axes.xaxis.set_major_formatter(PercentFormatter(1, decimals=3))

axes.set_title("Share of encounters where each drug was given (log scale)")
axes.text(
    0.99,
    0.19,
    "blue: >1% prescribed",
    transform=axes.transAxes,
    ha="right",
    color=BLUE,
    fontsize=12,
)
axes.text(
    0.99,
    0.12,
    "grey: 0–1% prescribed",
    transform=axes.transAxes,
    ha="right",
    color=GREY,
    fontsize=12,
)
axes.grid(axis="x")
axes.grid(axis="y", visible=False)
save_figure(fig, "10_medications")
plt.show()


# %% [markdown]
# As shown in the figure above, `examide` and `citoglipton` are never given. These columns can then be dropped, since they are constant throughout the data.
# Thirteen other medications are given in fewer than 1% of encounters. While too rare to learn from one by one, they can still be used as added information about the treatment. All other medications will have their own column.
#
# We can also check the effect on readmission rate of the number of medications given to each patient, as well as the dosage variations.
#

# %%
n_diabetes_medications = (
    (meds != "No").sum(axis=1).clip(upper=3).astype(str).replace("3", "3+")
)
n_dosage_changes = (
    meds.isin(["Up", "Down"]).sum(axis=1).clip(upper=2).astype(str).replace("2", "2+")
)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)

rate_bars(
    axes[0],
    not_excluded_discharges,
    n_diabetes_medications,
    title="Readmission rate by number of diabetes medications given",
)
rate_bars(
    axes[1],
    not_excluded_discharges,
    n_dosage_changes,
    highlight=["1", "2+"],
    title="Readmission rate by number of dosage changes",
)
for ax in axes:
    ax.set_xlabel("")
save_figure(fig, "10_medication_counts")
plt.show()

# %% [markdown]
# Comparing to the total readmission rate (11,4%), we can see that patients who were given only one type of medication tend to be readmitted more often than patients who did not need medication (probably lower risk cases, given that the hospice care and expired patients have already been removed from the data) and patients who needed more medication (perhaps the medication resolved the issue, or the disease was in too advanced of a state to allow for readmission).
#
# Moreover, changes in dosage clearly result in more readmissions. We suspect that this could be interpreted as a sign of **patient instability**, which could eventually translate to need for readmission in the near future.
#
# With all of this is mind, our data preprocessing decisions regarding the medicine columns will focus on reducing the number of columns, while simultaneously not losing information or important:
#
# - Drop `examide` and `citoglipton` column, since these are never prescribed (therefore are constant).
# - Maintain a column for each of the high prescription rate medications (>1%).
# - Create new `n_diabetes_medications` and `n_dosage_changes` columns, since those affect readmission rates.
# - Low prescription rate medication (<1%) will only be used in the `n_diabetes_medications` and `n_dosage_changes` columns. No dedicated column.
# - Make `change` and `diabetesMed` binary (Yes = 1, No = 0)
#

# %% [markdown]
# ### 11. Prior visits
#
# There are three columns in the dataset that give informations on visits from the patient in the previous year:
#
# - **number_outpatient**: visits as an outpatient (ambulatory)
# - **number_inpatient**: visits as an inpatient (staying in the hospital overnight)
# - **number_emergency**: emergency visits
#

# %%
visits = ["number_outpatient", "number_inpatient", "number_emergency"]

fig, axes = plt.subplots(1, 3, figsize=(19, 3.8), sharey=True)
for ax, col in zip(axes, visits):
    values = not_excluded_discharges[col].clip(upper=4).astype(str).replace("4", "4+")
    rate_bars(
        ax,
        not_excluded_discharges,
        values.rename(col),
        title=f"{col} ({(not_excluded_discharges[col] == 0).mean():.0%} are zero)",
    )
fig.suptitle(
    "<30-day readmission rate by prior visits", x=0.01, ha="left", fontweight="bold"
)
plt.tight_layout()
save_figure(fig, "11_prior_visits")
plt.show()

# %% [markdown]
# While the majority of encounters are from patients that had no prior visits in the last year (67-89%), it is clear that the readmission rate is heavily linked with the growing number of visits for **inpatients** and **emergency visits**.
#
# Knowing this, we will keep the columns as they are (to ensure that the information on the type of visit is not lost), and add a fourth column, `n_prior_visits`. This column is simply the sum of all values in the 3 original columns.
#

# %%
prior = not_excluded_discharges[visits].sum(axis=1).rename("n_prior_visits")

fig, axes = plt.subplots(figsize=(15, 4))
rate_bars(
    axes,
    not_excluded_discharges,
    prior.clip(upper=7).astype(str).replace("7", "7+"),
    title=f"<30-day readmission rate by n_prior_visits ({(prior == 0).mean():.0%} are zero)",
)
save_figure(fig, "11_prior_visits_sum")
plt.show()

# %% [markdown]
# ### 12. High-cardinality features: Medical specialty and payer code
#

# %%
medical_specialty_counts = (
    not_excluded_discharges["medical_specialty"]
    .fillna("Unknown")
    .value_counts(normalize=True)
)
payer_code_counts = (
    not_excluded_discharges["payer_code"].fillna("Unknown").value_counts(normalize=True)
)
print(
    "Number of different medical_specialty values: ", medical_specialty_counts.shape[0]
)
print("Number of different payer_code values: ", payer_code_counts.shape[0])

# %% [markdown]
# The `medical_specialty` feature can hold information regarding the patient encounter (the area of the health problem, especially when the ICD_9 diagnosis is missing), but it presents **73** possible values, way too many to one-hot encode. The same could be said for `payer_code`. Inspecting the `value_counts`, we can conclude that there are several values with a very low representation (<1%).
#

# %%
fig, axes = plt.subplots(figsize=(16, 4))

bar_colors = [
    BLUE if value >= config.MIN_CATEGORY_THRESHOLD else ORANGE
    for value in medical_specialty_counts
]

axes.bar(
    range(len(medical_specialty_counts)),
    medical_specialty_counts,
    color=bar_colors,
    width=0.8,
)
axes.set_yscale("log")
axes.axhline(config.MIN_CATEGORY_THRESHOLD, color=TEXT, linestyle="--", linewidth=1)
axes.text(
    len(medical_specialty_counts) - 1,
    config.MIN_CATEGORY_THRESHOLD * 1.15,
    "1%",
    ha="right",
    fontsize=9,
)
axes.yaxis.set_major_formatter(PercentFormatter(1, decimals=3))

rare = medical_specialty_counts < config.MIN_CATEGORY_THRESHOLD
sum_rare_specialties = medical_specialty_counts[rare].sum()
axes.set_title(
    f"medical_specialty: share of each of the {len(medical_specialty_counts)} values (log scale). "
    f"{rare.sum()} under {config.MIN_CATEGORY_THRESHOLD:.0%}, {sum_rare_specialties:.1%} of encounters in total"
)
axes.set_xlabel("specialties, most frequent first")

save_figure(fig, "12_specialty")
plt.show()


print(
    f"payer_code: {len(payer_code_counts)} values, {(payer_code_counts < config.MIN_CATEGORY_THRESHOLD).sum()} under 1%"
)


# %% [markdown]
# Knowing this, we will merge all categories with **less than 1%** representation into an `Infrequent` category. This category will also include unseen values in training, so that the model does not break.
#

# %% [markdown]
# ### 13. Numeric features and outliers
#
# In total, there are 8 numeric features in the **raw** dataset, given by the following `NUMERIC_FEATURES` array.
#

# %%
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))
for ax, col in zip(axes.flatten(), NUMERIC_FEATURES):
    values = not_excluded_discharges[col]
    skew = values.skew()
    ax.hist(values, bins=min(40, values.nunique()), color=BLUE, edgecolor="white")
    ax.axvline(values.quantile(0.99), color=TEXT, linestyle="--", linewidth=1)
    ax.set_title(f"{col}; skew = {skew:.2f}", fontsize=10)
fig.suptitle(
    "Numeric distributions (dashed line: 99th percentile)",
    x=0.01,
    ha="left",
    fontweight="bold",
)
plt.tight_layout()
save_figure(fig, "13_numeric_features")
plt.show()

not_excluded_discharges[NUMERIC_FEATURES].describe(percentiles=[0.01, 0.5, 0.99]).T

# %% [markdown]
# All values are non-negative, which checks out, but most features are right-skewed with a long tail. Regarding outliers, for example, one patient has 76 prior emergency visits, while 99% have 3 or fewer. Such extremes greatly influence distance based models. With this is mind, data preprocessing will focus on:
#
# - **Removing outliers**: Clip the counts at the 1 and 99% percentiles, to remove extreme cases such as the previous example. Exception to this is the age bands, as these would simply remove the first and last bands.
# - **Handling data skewness**: Apply the Yeo-Johnson transformation to the skewed features, since it handles zero correctly, unlike logarithmic transformations.
# - **Standartization**: Standardize, using a `StandardScaler`.
#
# This will only be done on the training rows, to avoid data leakage.
#

# %% [markdown]
# ### 14. Correlation between numeric features
#
# Through Spearman correlation, we are able to check if there are numeric features that correlate and can be dropped.
#

# %%
corr = not_excluded_discharges[NUMERIC_FEATURES].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(8, 6.5))
image = ax.imshow(np.ma.masked_where(mask, corr), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(NUMERIC_FEATURES)), NUMERIC_FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(NUMERIC_FEATURES)), NUMERIC_FEATURES)
for i in range(len(NUMERIC_FEATURES)):
    for j in range(i):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(image, ax=ax, shrink=0.7, label="Spearman rho")
ax.set_title("Spearman correlation, numeric features")
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
save_figure(fig, "14_correlation")
plt.show()

# %% [markdown]
# As seen above, no feature is very correlated to others. The most correlated pair is `num_medications/time_in_hospital`, which makes sense: more days in the hospital usually means more administrated medication.
#
# Nevertheless, a max rho of 0.46 is not enough to drop features. No numeric features will be dropped during data preprocessing.
#
