"""Read-only import of the supplied 50-column diabetes CSV format.

No dates are inferred from encounter IDs, row order, stay duration or source
outcome labels. All original cells remain strings, including '?' and 'None'.
"""

from dataclasses import dataclass
import csv
import hashlib
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

from audit_checks import EXPECTED_COLUMNS

INTEGER_COLUMNS = (
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
OUTCOMES = ("<30", ">30", "NO")


@dataclass
class HistoryDataset:
    source: str
    sha256: str
    rows: pd.DataFrame
    kind: str = "raw"

    @property
    def summary(self):
        if self.kind == "processed":
            labels = self.rows["readmitted_30d"].value_counts()
            label_counts = labels.to_dict()
            positive = int(label_counts.get("1", 0))
            negative = int(label_counts.get("0", 0))
            patients = (
                int(self.rows["patient_nbr"].nunique())
                if "patient_nbr" in self.rows.columns
                else None
            )
            return {
                "encounters": len(self.rows),
                "patients": patients,
                "<30": positive,
                ">30": 0,
                "NO": negative,
                "processed": True,
            }
        labels = self.rows["readmitted"].value_counts()
        return {
            "encounters": len(self.rows),
            "patients": self.rows["patient_nbr"].nunique(),
            **{label: int(labels.get(label, 0)) for label in OUTCOMES},
        }


def load_history(path):
    """Validate a replacement snapshot; never append or rewrite the source file."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Choose a .csv file in the diabetic_data.csv format.")
    content = path.read_bytes()
    try:
        decoded = content.decode("utf-8-sig")
        reader = csv.reader(StringIO(decoded), strict=True)
        header = next(reader, [])
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names.")
        if "readmitted_30d" in header:
            if len(header) < 2:
                raise ValueError("The processed CSV must contain features and readmitted_30d.")
            for number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV row {number} has {len(row)} cells; expected {len(header)}."
                    )
            frame = pd.read_csv(
                BytesIO(content), encoding="utf-8-sig", dtype=str, keep_default_na=False
            )
            if frame.empty:
                raise ValueError("The CSV has column names but no sample rows.")
            if not frame["readmitted_30d"].isin(("0", "1")).all():
                raise ValueError("readmitted_30d must contain only 0 or 1.")
            return HistoryDataset(
                path.name,
                hashlib.sha256(content).hexdigest(),
                frame,
                "processed",
            )
        if set(header) != set(EXPECTED_COLUMNS):
            missing = sorted(set(EXPECTED_COLUMNS) - set(header))
            extra = sorted(set(header) - set(EXPECTED_COLUMNS))
            raise ValueError(
                f"Expected the 50 supplied columns. Missing: {missing}; extra: {extra}."
            )
        for number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV row {number} has {len(row)} cells; expected {len(header)}."
                )
        frame = pd.read_csv(
            BytesIO(content), encoding="utf-8-sig", dtype=str, keep_default_na=False
        )
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError) as error:
        raise ValueError(
            "Cannot read this CSV; use UTF-8 and the supplied table format."
        ) from error
    if frame.empty:
        raise ValueError("The CSV has column names but no encounter rows.")
    # Reordered valid headers are accepted; stored fields have the source schema order.
    frame = frame[EXPECTED_COLUMNS]
    for column in INTEGER_COLUMNS:
        if not frame[column].str.fullmatch(r"[0-9]+").all():
            raise ValueError(
                f"Column {column} must contain nonnegative whole-number values."
            )
    if frame["encounter_id"].duplicated().any():
        raise ValueError("Duplicate encounter_id values need correction before import.")
    if not frame["readmitted"].isin(OUTCOMES).all():
        raise ValueError("readmitted must contain the unchanged labels <30, >30 or NO.")
    return HistoryDataset(path.name, hashlib.sha256(content).hexdigest(), frame)


def replace_history(path, current):
    """An invalid upload leaves the last valid snapshot available and labeled."""
    try:
        loaded = load_history(path)
    except (OSError, ValueError) as error:
        previous = (
            f"Still showing {current.source}."
            if current is not None
            else "No history loaded."
        )
        return current, f"Import rejected: {error} {previous}"
    return (
        loaded,
        f"Loaded {loaded.source}. This replaces the current snapshot; it does not add duplicate rows.",
    )
