"""Reproducible holdout evaluation and cross-validation orchestration."""

import hashlib
import json
import platform
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from car_price_prediction.config import ExperimentConfig
from car_price_prediction.data import FEATURES, TARGET, load_listings
from car_price_prediction.model import make_model
from car_price_prediction.plots import save_diagnostics


def run_experiment(
    config: ExperimentConfig,
    *,
    data_loader: Callable[[Path], pd.DataFrame] = load_listings,
    model_factory: Callable[[float], Pipeline] = make_model,
) -> dict[str, object]:
    """Split once, cross-validate only training data, then evaluate untouched holdout."""
    data = data_loader(config.data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data.loc[:, FEATURES], data[TARGET], test_size=config.test_size,
        random_state=config.random_state
    )
    if len(y_test) < 2:
        raise ValueError("Holdout must contain at least two rows to compute R²")
    if len(x_train) // config.cv_folds < 2:
        raise ValueError(
            "Each cross-validation fold needs at least two validation rows to compute R²"
        )

    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    validation = cross_validate(
        model_factory(config.alpha), x_train, y_train, cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}
    )

    model = model_factory(config.alpha)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    baseline = DummyRegressor(strategy="median")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)

    results: dict[str, object] = {
        "data_file": str(config.data_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_sha256": hashlib.sha256(config.data_path.read_bytes()).hexdigest(),
        "rows_after_exact_deduplication": len(data),
        "train_rows": len(x_train),
        "holdout_rows": len(x_test),
        "seed": config.random_state,
        "holdout_fraction": config.test_size,
        "cv_folds": config.cv_folds,
        "ridge_alpha": config.alpha,
        "cv_train_mae_mean_toman": float(-np.mean(validation["test_mae"])),
        "cv_train_mae_std_toman": float(np.std(-validation["test_mae"])),
        "cv_train_r2_mean": float(np.mean(validation["test_r2"])),
        "holdout_mae_toman": float(mean_absolute_error(y_test, predicted)),
        "holdout_r2": float(r2_score(y_test, predicted)),
        "median_baseline_holdout_mae_toman": float(
            mean_absolute_error(y_test, baseline_predictions)
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.output_dir / "model.joblib")
    (config.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame({
        "actual_toman": y_test.to_numpy(),
        "predicted_toman": predicted,
        "residual_toman": y_test.to_numpy() - predicted,
    }).to_csv(config.output_dir / "holdout_predictions.csv", index=False)
    save_diagnostics(y_test.to_numpy(), predicted, config.output_dir)
    return results
