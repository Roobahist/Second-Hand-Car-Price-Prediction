"""Validated parameters for experiments and reproducible outputs."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path = Path("Preprocessing/cleaned_data.csv")
    output_dir: Path = Path("artifacts")
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    alpha: float = 10.0

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if not isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
