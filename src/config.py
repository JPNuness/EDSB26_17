## Paths and settings used to simplify notebooks.

# Unifies variable names, ensuring that notebooks 
# remain consistent even when changing values or column names

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_env_file(env_file= PROJECT_ROOT / ".env"):
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())

# loads the data file names into the environment
load_env_file()

# Data files
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV = RAW_DATA_DIR / os.environ.get("RAW_DATA_FILE", "diabetic_data.csv")
IDS_CSV = RAW_DATA_DIR / os.environ.get("IDS_MAPPING_FILE", "IDS_mapping.csv")

# Generated figures and reports
GENERATED = PROJECT_ROOT / "reports" / "generated"