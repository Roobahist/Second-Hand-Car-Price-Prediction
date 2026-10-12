"""CLI parsing should remain stable across entry points."""

from car_price_prediction.cli import build_parser


def test_train_arguments() -> None:
    options = build_parser().parse_args(["train", "--folds", "3", "--seed", "21"])
    assert options.command == "train"
    assert options.folds == 3
    assert options.seed == 21
