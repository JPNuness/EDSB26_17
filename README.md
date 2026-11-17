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

On Windows, run the following in PowerShell from the project root instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Configure the raw data filenames

Before running a notebook, open `.env` and make sure both values match the exact CSV filenames in `data/raw`. The copied `.env.example` contains placeholders, so leaving them unchanged will cause a `FileNotFoundError`.

For the standard project datasets, `.env` should contain:

```dotenv
RAW_DATA_FILE=diabetic_data.csv
IDS_MAPPING_FILE=IDS_mapping.csv
```

If your files have different names, use those names instead. If you edit `.env` while a notebook is open, restart its kernel before rerunning cells so the updated values are loaded.

## Notebooks

| Notebook | Command |
|---|---|
| 1. Data visualization notebook (plots saved to `reports/generated/figures`) | `jupytext --to ipynb notebooks/data_visualization.py` |
| 2. Data preprocessing notebook | `jupytext --to ipynb notebooks/preprocessing.py` |

To save the notebooks as a `.py` file before commiting, run the same commands, but reversed.

**Example**: `jupytext --to py notebooks/data_visualization.ipynb`

## References
- List of ICD-9 codes (diagnosis groups): https://en.wikipedia.org/wiki/List_of_ICD-9_codes