import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from src.config import (
    BINARY,
    CATEGORICAL,
    MIN_CATEGORY_THRESHOLD,
    NUMERIC,
    ORDINAL,
    SKEWED,
    FINAL_FEATURES,
    TARGET,
)

# Builds the pipeline for data preprocessing
def build_preprocessor():
    preprocessor = ColumnTransformer(
        [
            (
                "skewed",
                Pipeline(
                    [
                        ("clip", Clipper()),
                        ("yeo_johnson", PowerTransformer(method="yeo-johnson"))
                    ]
                ),
                SKEWED
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("clip", Clipper()),
                        ("standardize", StandardScaler())
                    ]
                ),
                NUMERIC
            ),
            (
                "ordinal",
                Pipeline(
                    [
                        ("standardize", StandardScaler())
                    ]
                ),
                ORDINAL
            ),
            (
                "binary",
                "passthrough",
                BINARY
            ),
            (
                "categorical",
                OneHotEncoder(
                    min_frequency=MIN_CATEGORY_THRESHOLD,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False
                ),
                CATEGORICAL
            )
        ],
        verbose_feature_names_out=False
    )

    return preprocessor.set_output(transform="pandas")


# Custom transformer to clip numeric features by their 1% and 99% percentiles, excluding outliers
class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.01, high=0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_quantiles_ = X.quantile(self.low)
        self.upper_quantiles_ = X.quantile(self.high)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lower_quantiles_, self.upper_quantiles_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# Create a dataframe with the final features and correct target variable
def make_set(data: pd.DataFrame):
    return make_features(data), data[TARGET].astype(int)

# Create a dataframe with the final features 
def make_features(data: pd.DataFrame):
    return data[FINAL_FEATURES]