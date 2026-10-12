"""Command-line interface for preparing, training, and predicting."""

import argparse
import json
from pathlib import Path

from car_price_prediction.config import ExperimentConfig
from car_price_prediction.experiment import run_experiment
from car_price_prediction.model import load_model, predict_price
from car_price_prediction.preprocess import prepare_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="car-price", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Clean a historical raw CSV export")
    prep.add_argument("--raw", type=Path, default=Path("Crawl/final_result.csv"))
    prep.add_argument("--output", type=Path, default=Path("data/processed.csv"))
    prep.add_argument("--minimum-count", type=int, default=10)
    prep.add_argument("--upper-quantile", type=float, default=0.99)

    train = commands.add_parser("train", help="Train and evaluate leakage-safe Ridge model")
    train.add_argument("--data", type=Path, default=Path("Preprocessing/cleaned_data.csv"))
    train.add_argument("--output", type=Path, default=Path("artifacts"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--alpha", type=float, default=10.0)

    predict = commands.add_parser("predict", help="Predict a car's price in toman")
    predict.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    for feature in ("brand", "color", "gear", "body"):
        predict.add_argument(f"--{feature}", required=True)
    predict.add_argument("--city", required=True, dest="loc_1")
    predict.add_argument("--year", required=True, type=int)
    predict.add_argument("--usage", required=True, type=float)

    gui = commands.add_parser("gui", help="Launch the lightweight desktop estimator")
    gui.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    gui.add_argument("--data", type=Path, default=Path("Preprocessing/cleaned_data.csv"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        count = prepare_csv(args.raw, args.output, minimum_count=args.minimum_count,
                            upper_quantile=args.upper_quantile)
        print(f"Saved {count} rows to {args.output}")
    elif args.command == "train":
        config = ExperimentConfig(
            data_path=args.data, output_dir=args.output, random_state=args.seed,
            test_size=args.test_size, cv_folds=args.folds, alpha=args.alpha
        )
        print(json.dumps(run_experiment(config), indent=2, ensure_ascii=False))
    elif args.command == "predict":
        values = {feature: getattr(args, feature)
                  for feature in ("brand", "color", "gear", "body", "loc_1", "year", "usage")}
        value = predict_price(load_model(args.model), values)
        print(f"{value:,.0f} toman")
    elif args.command == "gui":
        from car_price_prediction.gui import launch_gui
        launch_gui(args.model, args.data)


if __name__ == "__main__":
    main()
