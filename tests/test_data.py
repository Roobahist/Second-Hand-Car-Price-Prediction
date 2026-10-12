"""Contracts for the historical listing schema and feature inputs."""

import pandas as pd
import pytest

from car_price_prediction.data import load_listings, validate_listings
from car_price_prediction.model import validate_features


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "brand": f"brand-{index % 4}", "color": f"color-{index % 3}",
            "gear": "manual" if index % 2 else "automatic",
            "year": 1380 + index, "body": "clean" if index % 2 else "repainted",
            "loc_1": f"city-{index % 3}", "usage": 1000 + index * 500,
            "price": 10_000_000 + index * 2_000_000,
        }
        for index in range(28)
    ])


def test_validates_schema_and_deduplicates(sample_data: pd.DataFrame) -> None:
    doubled = pd.concat([sample_data, sample_data.iloc[:1]], ignore_index=True)
    assert len(validate_listings(doubled)) == len(sample_data)
    with pytest.raises(ValueError, match="Missing"):
        validate_listings(sample_data.drop(columns="price"))


def test_rejects_missing_invalid_and_negative_data(sample_data: pd.DataFrame) -> None:
    sample_data.loc[0, "usage"] = -2
    with pytest.raises(ValueError, match="Mileage"):
        validate_listings(sample_data)
    sample_data.loc[0, "usage"] = "unknown"
    with pytest.raises(ValueError, match="missing or nonnumeric"):
        validate_listings(sample_data)


def test_ignores_legacy_csv_index(tmp_path, sample_data: pd.DataFrame) -> None:
    file = tmp_path / "listings.csv"
    sample_data.to_csv(file, index=True)
    assert list(load_listings(file).columns) == [
        "brand", "color", "gear", "year", "body", "loc_1", "usage", "price"
    ]


def test_prediction_contract() -> None:
    row = {
        "brand": "car", "color": "white", "gear": "manual", "year": 1395,
        "body": "clean", "loc_1": "Tehran", "usage": 40000
    }
    assert validate_features(row).iloc[0]["year"] == 1395
    with pytest.raises(ValueError, match="Mileage"):
        validate_features({**row, "usage": -1})
    with pytest.raises(ValueError, match="Missing"):
        validate_features({k: v for k, v in row.items() if k != "body"})
