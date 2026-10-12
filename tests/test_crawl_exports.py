"""Offline checks for crawler pagination and CSV assembly."""

import pandas as pd
import pytest

from car_price_prediction.crawl_exports import combine_csv_exports
from car_price_prediction.crawler_urls import listing_urls


def test_listing_urls_are_unique_and_bounded() -> None:
    links = listing_urls(["tehran", "karaj", "tehran"], "pride/131", pages=3)
    assert len(links) == 6
    assert len(set(links)) == 6
    assert links[0].endswith("pride/131?page=1")
    with pytest.raises(ValueError, match="pages"):
        listing_urls(pages=0)
    with pytest.raises(ValueError, match="car_model"):
        listing_urls(car_model="../bad")


def test_combine_exports_is_deterministic(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"brand": ["B"], "price": [20]}).to_csv(source / "b.csv", index=False)
    pd.DataFrame({"brand": ["A"], "price": [10]}).to_csv(source / "a.csv", index=False)
    out = source / "combined.csv"
    assert combine_csv_exports(source, out) == 2
    assert list(pd.read_csv(out)["brand"]) == ["A", "B"]
    assert combine_csv_exports(source, out) == 2
    assert list(pd.read_csv(out)["brand"]) == ["A", "B"]


def test_combine_rejects_mismatched_schema(tmp_path) -> None:
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"y": [2]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="Column mismatch"):
        combine_csv_exports(tmp_path, tmp_path / "out.csv")
