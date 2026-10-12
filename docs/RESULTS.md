# Verified maintained-model benchmark

This page records measurements from
[GitHub Actions run 38300441780](https://github.com/Roobahist/Second-Hand-Car-Price-Prediction/actions/runs/38300441780),
executed on October 12, 2026. All checks passed on
commit `cf23f4ad294dc115861851f2a19d416c59241ef0`.

Command:

```bash
car-price train --folds 3 --output artifacts
```

The input was `Preprocessing/cleaned_data.csv`, SHA-256
`f70418b4a282fa4b00534b3d7c5c2ca2ce41f559ce5f947d2c9cea4594a7290d`.
This run used a seeded 80/20 train/holdout split (seed 42) and Ridge alpha 10.

The recorded software environment was Python 3.11.17, NumPy 2.4.6,
pandas 2.3.3, and scikit-learn 1.9.1. These versions are stored in
`metrics.json` alongside the dataset checksum and experiment parameters.

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
