"""Checks for the deterministic recreation of historical data filters."""

import pandas as pd
import pytest

from car_price_prediction.preprocess import prepare_listings


def test_prepares_city_brand_and_numeric_fields() -> None:
    raw = pd.DataFrame([
        {
            "body": "clean", "brand": "Peugeot 405 GLX-petrol",
            "cit": "تهران، مرکز", "color": "سفید", "gear": "manual",
            "price": 20_000_000 + index * 500_000,
            "usage": index * 1000, "year": 1390 + index % 3,
        }
        for index in range(20)
    ])
    cleaned = prepare_listings(raw, minimum_count=1, upper_quantile=0.99)
    assert not cleaned.empty
    assert set(cleaned["loc_1"]) == {"تهران"}
    assert set(cleaned["brand"]) == {"پژو 405 GLX بنزینی"}
    assert cleaned["price"].min() >= 1_000_000


def test_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        prepare_listings(pd.DataFrame(), minimum_count=0)
