## Dataset related helpers to simplify notebooks

import numpy as np
import pandas as pd

from src.config import RAW_CSV

def read_raw(path=RAW_CSV):
    return pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)