"""Compatibility wrapper for combining per-model crawler exports."""

import argparse
from pathlib import Path

from car_price_prediction.crawl_exports import combine_csv_exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parent = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=parent / "New Folder")
    parser.add_argument("--output", type=Path, default=parent / "combined_csv.csv")
    options = parser.parse_args()
    count = combine_csv_exports(options.input_dir, options.output)
    print(f"Combined {count} files into {options.output}")


if __name__ == "__main__":
    main()
