# EDSB26 - Group 17

## Diabetes - Hospital Readmission

## Project Layout

```
data/raw          diabetic_data.csv, IDS_mapping.csv: Raw datasets
notebooks         All notebooks
src/config.py     Path rules and column list
src/dataset.py    Dataset cleaning rules and feature engineering
reports           Our reports. Reports/generated is written by the code
```
## Setup

In the terminal, navigate to the project root folder and run the following commands to setup the project. These will:

1. Create a virtual environment
2. Activate it
3. Install all packages required by the project
4. Create the env file required to indicate the data file names.

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

## Notebooks

| Notebook | Command |
|---|---|
| 1. Data visualization notebook (plots saved to `reports/generated/figures`) | `jupytext --to ipynb notebooks/data_visualization.py` |

To save the notebooks as a `.py` file before commiting, run the same commands, but reversed.

**Example**: `jupytext --to py notebooks/data_visualization.ipynb`

## References
- List of ICD-9 codes (diagnosis groups): https://en.wikipedia.org/wiki/List_of_ICD-9_codes