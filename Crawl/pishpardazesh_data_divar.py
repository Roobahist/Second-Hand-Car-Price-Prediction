"""Compatibility entrypoint for the historical data cleaning stage."""

import argparse
from pathlib import Path

from car_price_prediction.preprocess import prepare_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("Crawl/divar_crawler/combined_csv.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed.csv"))
    parser.add_argument("--minimum-count", type=int, default=10)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    options = parser.parse_args()
    count = prepare_csv(
        options.raw, options.output, minimum_count=options.minimum_count,
        upper_quantile=options.upper_quantile
    )
    print(f"Prepared {count} rows into {options.output}")


if __name__ == "__main__":
    main()
