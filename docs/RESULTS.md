# Verified maintained-model benchmark

This page records measurements from
[GitHub Actions run 38236856529](https://github.com/Roobahist/Second-Hand-Car-Price-Prediction/actions/runs/38236856529),
executed on October 12, 2026. All checks passed on
commit `a22694be47ab83ee809deefd500ad884d6a9f079`.

Command:

```bash
car-price train --folds 3 --output artifacts
```

The input was `Preprocessing/cleaned_data.csv`, SHA-256
`f70418b4a282fa4b00534b3d7c5c2ca2ce41f559ce5f947d2c9cea4594a7290d`.
This run used a seeded 80/20 train/holdout split (seed 42) and Ridge alpha 10.

| Measurement | Observed value |
| --- | ---: |
| Input rows | 11,372 |
| Unique exact rows after deduplication | 11,116 |
| Train records | 8,892 |
| Holdout records | 2,224 |
| Three-fold training CV mean MAE | 6,134,245.83 toman |
| Three-fold training CV mean R² | 0.9111 |
| Holdout MAE | **5,894,974.47 toman** |
| Holdout R² | **0.9227** |
| Holdout median-baseline MAE | 23,164,170.41 toman |

The action uploaded `historical-baseline`, containing its actual
`metrics.json`, holdout prediction CSV, and two generated PNG diagnostic
figures. See the linked workflow run's **Artifacts** section to inspect them.

The figures were created from the real holdout predictions; they are not
drawn from the copied 2022 notebook metrics. This benchmark still inherits
the original dataset's age, preprocessing selection bias, and possible
near-duplicate listings. Do not interpret the holdout figures as a market
forecast or compare them directly to the original network's 5-fold numbers.
