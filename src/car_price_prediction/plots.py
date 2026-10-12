"""Saved, headless diagnostics derived from holdout predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_diagnostics(actual: np.ndarray, predicted: np.ndarray, directory: Path) -> None:
    """Create separate, legible prediction and residual plots."""
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(actual / 1_000_000, predicted / 1_000_000, s=12, alpha=0.35)
    ax.plot([lower / 1_000_000, upper / 1_000_000],
            [lower / 1_000_000, upper / 1_000_000], linestyle="--", color="black")
    ax.set(xlabel="Actual price (million toman)", ylabel="Predicted price (million toman)",
           title="Holdout predictions")
    fig.tight_layout()
    fig.savefig(directory / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(predicted / 1_000_000, (actual - predicted) / 1_000_000,
               s=12, alpha=0.35)
    ax.axhline(0, linestyle="--", color="black")
    ax.set(xlabel="Predicted price (million toman)", ylabel="Residual (million toman)",
           title="Holdout residuals")
    fig.tight_layout()
    fig.savefig(directory / "residuals.png", dpi=160)
    plt.close(fig)
