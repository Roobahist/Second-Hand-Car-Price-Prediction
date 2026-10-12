"""Models and inference contracts shared by training, CLI, and desktop UI."""

from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from car_price_prediction.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def make_model(alpha: float = 10.0) -> Pipeline:
    """Build a model that fits all preprocessing inside each training fold."""
    transform = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ]
    )
    return Pipeline([("features", transform), ("regressor", Ridge(alpha=alpha, solver="lsqr"))])


def validate_features(values: Mapping[str, object]) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(values))
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")
    row = {name: values[name] for name in FEATURES}
    for name in CATEGORICAL_FEATURES:
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
        row[name] = row[name].strip()
    for name in NUMERIC_FEATURES:
        try:
            row[name] = float(row[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not pd.notna(row[name]) or not -float("inf") < row[name] < float("inf"):
            raise ValueError(f"{name} must be finite")
    if row["usage"] < 0 or row["year"] <= 0:
        raise ValueError("Mileage must be nonnegative and year must be positive")
    return pd.DataFrame([row], columns=FEATURES)


def load_model(path: Path) -> Pipeline:
    """Load a local artifact from a trusted training run only.

    joblib uses pickle internally and must not be used for untrusted model files.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run car-price train first.")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError("Expected a scikit-learn Pipeline artifact")
    return model


def predict_price(model: Pipeline, values: Mapping[str, object]) -> float:
    """Predict price in Iranian toman without masking negative predictions."""
    return float(model.predict(validate_features(values))[0])
