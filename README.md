# Second-Hand Car Price Prediction

A reproducible machine-learning study of **historical Iranian used-car listings** collected
from Divar. The original 2022 project explored linear regression and a neural network, and
included data collection, notebooks, a Persian Tkinter interface, and a project report.

This repository now provides a separate, testable scikit-learn training pipeline and a
small desktop estimator. It preserves the historical implementation and its datasets
for research context rather than silently replacing the original experimental results.

> **Not a current pricing service.** The sample is dated and limited to selected Iranian
> cities and car models. Predictions are in Iranian **toman** and must not be treated as
> live valuations.

## Research snapshot

| Historical model | Five-fold test MAE | Mean five-fold test R² |
| --- | ---: | ---: |
| Linear regression | 6.05 million toman | 0.9171 |
| Deep neural network | 4.52 million toman | 0.9288 |

These numbers are copied from the saved outputs in `Analysis/MainAnalysis.ipynb`, not
new measurements. In particular, the original neural-network workflow standardized
the full dataset before splitting folds, creating **validation leakage**. It also reused
fold 1 in the residual visualization loop. See [research notes](docs/RESEARCH.md).
The new Ridge pipeline uses a different evaluation protocol, so results are **not**
directly comparable to this table. See the [verified maintained-model benchmark](docs/RESULTS.md).

## Quick start

Install Python 3.11 or newer and run the following from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
car-price train
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The training command uses `Preprocessing/cleaned_data.csv`, performs a seeded
80/20 train/holdout split, evaluates five shuffled folds **within training only**,
then fits one model and evaluates the held-out records exactly once.

Training writes `artifacts/model.joblib`, `metrics.json`,
`holdout_predictions.csv`, `actual_vs_predicted.png` and `residuals.png`.
Generated results are ignored by Git. Training is reproducible on the same input
file and library versions; record the data SHA-256 and dependency versions. Use the checked-in `uv.lock`
to preserve resolved versions across machines.
R² evaluation requires at least two holdout observations and at least two
validation observations per cross-validation fold. The command rejects splits
that cannot produce meaningful R² values.

For a different seed, dataset, fold count or regularization strength:

```bash
uv run --locked car-price train --data Preprocessing/cleaned_data.csv --output artifacts --seed 42 --folds 5 --alpha 10
```

Predict from the same fitted preprocessing and model artifact:

```bash
uv run --locked car-price predict --model artifacts/model.joblib --brand "پژو 206 تیپ ۲" --color "سفید" --gear "دنده‌ای" --year 1394 --body "بدون رنگ" --city "تهران" --usage 80000
```

Or launch the desktop interface on a computer with Tkinter and a graphical desktop:

```bash
uv run --locked car-price gui --model artifacts/model.joblib
```

Only load model files from **trusted** sources. scikit-learn/joblib model artifacts
rely on Python pickle serialization.

## Recreate a cleaned dataset from the crawl

```bash
uv run --locked car-price prepare --raw Crawl/final_result.csv --output data/processed.csv
uv run --locked car-price train --data data/processed.csv --output artifacts/derived
```

To reconstruct the original per-model crawler export, use
`uv run --locked python Crawl/divar_crawler/concate_cars_csv.py` after installing the package.
The optional spider and its constraints are documented in
[Crawl/README.md](Crawl/README.md).

The preparation command exposes the historical rare-category and 99th-percentile
trimming rules and removes exact duplicates. It **does not guarantee a byte-for-byte
reconstruction** of the legacy `cleaned_data.csv`, whose notebook shuffled data
without a fixed seed. Check preprocessing bias before using a cleaned dataset.

## Directory overview

| Location | Purpose |
| --- | --- |
| `src/car_price_prediction/` | Validated ingestion, training, inference, plotting, CLI, desktop UI |
| `tests/` | Isolated, synthetic regression tests |
| `docs/` | Data and experiment caveats, architecture and maintenance |
| `Preprocessing/` | Original prepared CSV files and historical notebook |
| `Analysis/` | Original regression and neural-network experiments |
| `Crawl/` | Configurable opt-in Scrapy crawler, historical exports and compatibility scripts |
| `Gui/` | Original Tkinter GUI, icon and legacy neural-network weights |
| `report.pdf` | Original research report, retained unchanged |

The old GUI remains as an archival reference. It has machine-specific paths,
hardcoded linear-regression weights, and an outdated neural-network dependency;
use `car-price gui` for a newly trained Ridge model.

## Development

```bash
uv sync --locked
uv run --locked ruff check src tests
uv run --locked pytest -q
```

CI runs these checks plus a full training job on the committed cleaned CSV. The
artifact contains metrics and the newly generated diagnostic figures. Docker is
not included because this project is primarily a local CLI and desktop research
application. The Scrapy crawler was not validated against today's Divar site and
is not executed by CI. Respect the site's terms and robots rules before attempting
new collection.

See [architecture](docs/ARCHITECTURE.md) and [experimental caveats](docs/RESEARCH.md).
No software or dataset license is specified by the historical repository.
