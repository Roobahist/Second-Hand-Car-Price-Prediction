"""Read and validate the curated historical listing data."""

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ("brand", "color", "gear", "body", "loc_1")
NUMERIC_FEATURES = ("year", "usage")
FEATURES = ("brand", "color", "gear", "year", "body", "loc_1", "usage")
TARGET = "price"


def validate_listings(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy without exact duplicate listings.

    Duplicate records can contaminate random train/test splits. Listing identifiers
    were not retained, so distinct postings describing the same car cannot be detected.
    """
    required = set(FEATURES) | {TARGET}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = data.loc[:, [*FEATURES, TARGET]].copy()
    for column in (*NUMERIC_FEATURES, TARGET):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype("string").str.strip()
        result[column] = result[column].replace("", pd.NA)

    if result.isna().any().any():
        raise ValueError("Data contains missing or nonnumeric required values")
    if not np.isfinite(result.loc[:, [*NUMERIC_FEATURES, TARGET]].to_numpy()).all():
        raise ValueError("Data contains infinite numeric values")
    if (result[TARGET] <= 0).any():
        raise ValueError("Prices must be positive")
    if (result["usage"] < 0).any():
        raise ValueError("Mileage cannot be negative")
    if (result["year"] <= 0).any():
        raise ValueError("Production years must be positive")
    return result.drop_duplicates().reset_index(drop=True)


def load_listings(path: Path) -> pd.DataFrame:
    """Load the cleaned historical dataset, independent of the current working directory."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return validate_listings(pd.read_csv(path))
