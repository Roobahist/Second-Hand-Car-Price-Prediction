"""Repeat the documented historical listing filters with explicit parameters."""

from pathlib import Path

import pandas as pd

from car_price_prediction.data import CATEGORICAL_FEATURES, FEATURES, TARGET, validate_listings

RAW_COLUMNS = ("body", "brand", "cit", "color", "gear", "price", "usage", "year")
BRAND_RENAMES = {"Peugeot 405 GLX-petrol": "پژو 405 GLX بنزینی"}


def prepare_listings(
    raw: pd.DataFrame, *, minimum_count: int = 10, upper_quantile: float = 0.99
) -> pd.DataFrame:
    """Transform the crawl export into model-ready observations.

    The 2022 notebooks also removed rare brand/color/year values and the top
    one percent of price/mileage. Those choices are replicated as parameters,
    not endorsed as generally suitable data-cleaning rules.
    """
    if minimum_count < 1 or not 0 < upper_quantile < 1:
        raise ValueError("Invalid minimum_count or upper_quantile")
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {', '.join(missing)}")

    data = raw.loc[:, RAW_COLUMNS].copy()
    data["loc_1"] = data["cit"].astype("string").str.split("،", n=1).str[0].str.strip()
    data["brand"] = data["brand"].replace(BRAND_RENAMES)
    for field in (TARGET, "usage", "year"):
        data[field] = pd.to_numeric(data[field], errors="coerce")
    data = data.dropna(subset=[*FEATURES, TARGET])
    data = data.loc[(data[TARGET] >= 1_000_000) & (data[TARGET] % 1_000 == 0)]
    data = data.loc[(data["usage"] >= 0) & (data["year"] > 0)]

    for field in ("brand", "color", "year"):
        counts = data[field].value_counts()
        data = data.loc[data[field].map(counts) >= minimum_count]
    for field in (TARGET, "usage"):
        data = data.loc[data[field] < data[field].quantile(upper_quantile)]
    return validate_listings(data)


def prepare_csv(
    raw_path: Path,
    output_path: Path,
    *,
    minimum_count: int = 10,
    upper_quantile: float = 0.99,
) -> int:
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    prepared = prepare_listings(
        pd.read_csv(raw_path), minimum_count=minimum_count, upper_quantile=upper_quantile
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    return len(prepared)
