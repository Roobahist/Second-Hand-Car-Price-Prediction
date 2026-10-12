"""End-to-end pipeline smoke tests on small, synthetic fixtures."""

import json

import pandas as pd
import pytest

from car_price_prediction.config import ExperimentConfig
from car_price_prediction.experiment import run_experiment
from car_price_prediction.model import load_model, predict_price


def test_training_outputs_and_reproducibility(tmp_path) -> None:
    rows = []
    for index in range(50):
        rows.append({
            "brand": f"model-{index % 5}", "color": f"color-{index % 3}",
            "gear": "manual" if index % 2 else "automatic",
            "year": 1380 + index % 20, "body": "clean" if index % 2 else "painted",
            "loc_1": f"city-{index % 4}", "usage": index * 2500,
            "price": 20_000_000 + (index % 5) * 5_000_000 + index * 130_000,
        })
    dataset = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    first = ExperimentConfig(
        data_path=dataset, output_dir=tmp_path / "run1",
        cv_folds=3, random_state=7
    )
    second = ExperimentConfig(
        data_path=dataset, output_dir=tmp_path / "run2",
        cv_folds=3, random_state=7
    )
    first_metrics = run_experiment(first)
    second_metrics = run_experiment(second)

    assert first_metrics == second_metrics
    assert first_metrics["holdout_rows"] > 0
    assert first_metrics["median_baseline_holdout_mae_toman"] >= 0
    assert json.loads((first.output_dir / "metrics.json").read_text()) == first_metrics
    for name in ("model.joblib", "holdout_predictions.csv",
                 "actual_vs_predicted.png", "residuals.png"):
        assert (first.output_dir / name).is_file()
    model = load_model(first.output_dir / "model.joblib")
    assert isinstance(predict_price(model, rows[0]), float)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -1.0])
def test_rejects_invalid_regularization(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be finite and non-negative"):
        ExperimentConfig(alpha=alpha)


@pytest.mark.parametrize(
    ("test_size", "cv_folds", "expected"),
    [
        (0.05, 3, "Holdout must contain at least two rows"),
        (0.2, 5, "Each cross-validation fold needs at least two validation rows"),
    ],
)
def test_rejects_splits_that_cannot_compute_r_squared(
    tmp_path, test_size: float, cv_folds: int, expected: str
) -> None:
    data = pd.DataFrame([
        {
            "brand": f"model-{i}", "color": "white", "gear": "manual",
            "year": 1380 + i, "body": "clean", "loc_1": "Tehran",
            "usage": 1_000 * i, "price": 5_000_000 + i * 2_000_000,
        }
        for i in range(10)
    ])
    path = tmp_path / "small.csv"
    data.to_csv(path, index=False)
    config = ExperimentConfig(
        data_path=path, output_dir=tmp_path / "output",
        cv_folds=cv_folds, test_size=test_size,
    )
    with pytest.raises(ValueError, match=expected):
        run_experiment(config)
