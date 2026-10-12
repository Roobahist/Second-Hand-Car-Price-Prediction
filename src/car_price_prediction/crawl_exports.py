"""Offline, reproducible operations on historical crawler exports."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def combine_csv_exports(
    directory: Path,
    output_path: Path,
    *,
    reader: Callable[[Path], pd.DataFrame] = pd.read_csv,
) -> int:
    """Combine source CSV files in a stable order without changing process CWD."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Export directory not found: {directory}")
    output_resolved = output_path.resolve()
    files = sorted(
        path for path in directory.glob("*.csv")
        if path.is_file() and path.resolve() != output_resolved
    )
    if not files:
        raise ValueError(f"No input CSV files found in {directory}")

    tables = [reader(path) for path in files]
    reference = list(tables[0].columns)
    for path, table in zip(files[1:], tables[1:], strict=True):
        if list(table.columns) != reference:
            raise ValueError(f"Column mismatch in {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(files)
