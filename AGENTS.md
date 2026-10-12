# Contributor and coding-agent guide

This is a historical used-car pricing research repository. Preserve original
notebooks, CSV snapshots, project report and trained weights unless a change is
explicitly justified. Never present notebook outputs as newly reproduced.

- New application code belongs in `src/car_price_prediction/`.
- Entrypoints live in `cli.py`; orchestrate training through `run_experiment`.
- Keep data loading, preprocessing, estimators and plots independently testable.
- Inject data loaders and model factories at meaningful boundaries. Avoid
  machine-specific paths, credentials and provider assumptions.
- Fit encoders and scalers inside the cross-validation pipeline, never on
  training and validation data combined.
- Preserve the untouched holdout split for final evaluation.
- Use the original Persian column values without recoding them manually.
- Reject missing and implausible inputs explicitly; do not force negative
  predictions positive with `abs`.
- Tests use synthetic fixtures, not external services or sites.
- To verify: `uv sync --locked`, `uv run --locked ruff check src tests`,
  `uv run --locked pytest -q`, `uv run --locked car-price train --folds 3`.
- Keep `uv.lock` committed and update it with `uv lock` when dependencies change.
- Generated models and plots are stored in ignored `artifacts/`.
- Do not load untrusted `.joblib` files; pickle can execute arbitrary code.
- Legacy scraper selectors have not been checked against the current website.
- Keep crawler URL generation separate from parse callbacks to avoid recursive pagination.
- Do not re-license historical datasets or alter Git history automatically.
