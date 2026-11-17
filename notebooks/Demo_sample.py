# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Diabetes: Demo encounter sample
#
# This notebook creates a reproducible batch of the configured number of encounters from the processed test set.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import assign_split, sample_encounters

pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.3f}".format)

# %% [markdown]
# ### 1. Load the test set

# %%
test = pd.read_csv(config.TEST_CSV)
print(f"Test set: {len(test):,} encounters")

# %% [markdown]
# ### 2. Create the configured demo batch

# %%
demo_batch = sample_encounters(test, n=config.SAMPLE_SIZE)
demo_batch.to_csv(config.PROCESSED_DATA_DIR / "test_sample.csv", index=False)

print(f"Demo batch: {len(demo_batch):,} encounters")
print("Saved to:", config.PROCESSED_DATA_DIR / "test_sample.csv")
demo_batch.head()

# %% [markdown]
# ### 3. Demographic overview
#
# The processed test CSV contains transformed features, so demographics are loaded from the clean test split to retain the original age bands, race, and gender.

# %%
clean = pd.read_csv(config.CLEAN_DATA_DIR / "clean.csv")
clean["split"] = assign_split(clean)
raw_test = clean[clean["split"] == "test"].copy()
raw_demo = sample_encounters(raw_test, n=config.SAMPLE_SIZE)

demo_demographics = pd.DataFrame(
    {
        "age": raw_demo["age"],
        "race": raw_demo["race"].fillna("Unknown"),
        "gender": raw_demo["gender"].replace("Unknown/Invalid", "Unknown"),
        "target": raw_demo[config.TARGET],
    }
)

def rate_bars(
    axis,
    data: pd.DataFrame,
    by,
    overall,
    highlight=(),
    sort=False,
    title=None,
    categories=None,
):
    rates = data.groupby(by, observed=True)["target"].agg(rate="mean", n="size")
    if categories is not None:
        rates = rates.reindex(categories)
        rates["rate"] = rates["rate"].fillna(0)
        rates["n"] = rates["n"].fillna(0).astype(int)
    if sort:
        rates = rates.sort_values("rate")

    labels = rates.index.astype(str)
    colors = [
        "#eb6834" if label in set(map(str, highlight)) else "#2a78d6"
        for label in rates.index
    ]
    axis.bar(labels, rates["rate"], color=colors, width=0.7)
    axis.set_xticks(
        range(len(rates)),
        [f"{label}\n(n = {n:,})" for label, n in zip(labels, rates["n"])],
        fontsize=8,
    )
    axis.axhline(overall, color="#0b0b0b", linestyle="--", linewidth=1)
    axis.annotate(
        f"all: {overall:.1%}",
        (0, overall),
        xycoords=("axes fraction", "data"),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=9,
        color="#0b0b0b",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.set_title(title or f"30-day readmission rate by {by}")
    return rates


full_test_rate = test[config.TARGET].mean()
AGE_BANDS = [f"[{age}-{age + 10})" for age in range(0, 100, 10)]
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 4),
    gridspec_kw={"width_ratios": [3, 2, 1.5]},
)

ages = rate_bars(
    axes[0],
    demo_demographics,
    "age",
    overall=full_test_rate,
    title="Readmission rate by age band",
    categories=AGE_BANDS,
)
axes[0].set_xlabel("")

rate_bars(
    axes[1],
    demo_demographics,
    demo_demographics["race"].fillna("?"),
    overall=full_test_rate,
    highlight=["?"],
    sort=True,
    title="Readmission rate by race",
)

rate_bars(
    axes[2],
    demo_demographics,
    demo_demographics["gender"].fillna("Unknown"),
    overall=full_test_rate,
    highlight=["Unknown"],
    sort=True,
    title="Readmission rate by gender",
)

for axis in axes:
    axis.set_ylabel("30-day readmission rate")

plt.suptitle(
    f"Demographics in the {config.SAMPLE_SIZE}-encounter demo sample "
    f"(sample target rate: {demo_demographics['target'].mean():.1%})",
    fontsize=14,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Target distribution

# %%
demo_demographics["target"].value_counts().sort_index().rename(
    index={0: "No 30-day readmission", 1: "30-day readmission"}
).plot(kind="bar", color=["#b4b2a9", "#eb6834"], edgecolor="white")
plt.title("30-day readmission outcome")
plt.ylabel("Encounters")
plt.xticks(rotation=0)
plt.show()

print(
    f"30-day readmission rate: "
    f"{demo_demographics['target'].mean():.1%}"
)
