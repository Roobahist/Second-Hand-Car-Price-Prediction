# Dataset, historical results and validity notes

## What is actually versioned

The historical cleaned dataset is `Preprocessing/cleaned_data.csv` with
11,372 rows and eight modeling columns. The raw filtered crawl is
`Crawl/final_result.csv`. The listings came from selected Divar car models and
Iranian cities and were collected before the original 2022 upload. The dataset
is not representative of a current vehicle market.

The historical preparation notebook drops rare model/color/year values,
excludes prices without thousands-place rounding, removes observations above
the 99th percentile of price and mileage, and shuffles without a recorded seed.
These choices can affect representativeness and reported error.

## Results preserved from saved notebook output

`Analysis/MainAnalysis.ipynb` reports:

| Model | Mean training MAE (million toman) | Mean held-out-fold MAE (million toman) | Mean held-out-fold R² |
| --- | ---: | ---: | ---: |
| Linear regression | 6.00 | 6.05 | 0.9171185054 |
| Neural network | 3.54 | 4.52 | 0.9287756357 |

These are historical notebook outputs, **not** fresh runs or new benchmark
results. The notebook uses a five-fold analysis on 11,372 rows and a deep
network with 2,757,951 trainable parameters.

### Known methodological limitations

1. The neural-network experiment calls `StandardScaler.fit_transform` on
   all encoded observations before defining train and test folds. Cross-fold
   information therefore enters the scaling statistics.
2. The label-encoded categories are scaled as if their integer labels were
   continuous and ordered.
3. The neural-network residual visualization uses fold index zero in every
   loop iteration rather than the corresponding fold.
4. The notebook computes a purported best-validation-loss table using loop
   variable `j` after a minimum-search loop, so those entries reflect the
   final epoch instead of the selected best epoch.
5. Historical preprocessing uses target-aware percentile trimming before
   model evaluation. The new workflow uses the *already cleaned* snapshot;
   it cannot undo this earlier selection bias.
6. The data has no stable listing identifiers. Exact duplicates can be
   removed, but near-duplicates and repostings cannot be fully separated
   into independent train/test groups.

For these reasons, historical metrics should not be promoted as independently
reproduced or directly compared with the new baseline.

## Maintained reproducible baseline

`car-price train` splits a validated version of the historical dataset once,
uses shuffled K-fold cross-validation **within the training partition** and
fits one `ColumnTransformer` + `Ridge` estimator on that partition. It then
reports the untouched holdout MAE and R² and a median-price baseline. All
transformers are fitted only to training data in each fold.

The new model intentionally differs from the historical regression and network.
Its metrics, split, hyperparameters, data hash and diagnostic plots are written
to `artifacts/` by the actual run, never inserted into this document as
unverified claims. Verify results in GitHub Actions or locally.
