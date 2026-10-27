from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


class DSLParseError(ValueError):
    pass


REQUIRED_DSL_FIELDS = {
    "task",
    "split",
    "seed",
    "preprocess_fit",
    "preprocess_apply",
    "tune_on",
    "metric",
    "cv",
    "cv_preprocess",
    "augment_on",
    "dataset_kind",
    "group_split",
    "time_shuffle",
}


def parse_dsl(text: str) -> dict[str, str]:
    """Parse the line-oriented experiment DSL into normalized fields."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines or not lines[0].startswith("EXPERIMENT "):
        raise DSLParseError("The first non-empty line must be 'EXPERIMENT <id>'")

    result = {"id": lines[0].split(maxsplit=1)[1].strip()}
    if not result["id"]:
        raise DSLParseError("Experiment identifier must not be empty")

    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DSLParseError(f"Line {line_number} must contain a key and value")
        key, value = parts[0].lower(), parts[1].strip()
        if key in result:
            raise DSLParseError(f"Duplicate field {key!r} on line {line_number}")
        if not value:
            raise DSLParseError(f"Field {key!r} must not be empty")
        result[key] = value

    missing = sorted(REQUIRED_DSL_FIELDS - result.keys())
    if missing:
        raise DSLParseError(f"Missing required fields: {', '.join(missing)}")
    return result


def validate_row(row: Mapping[str, object]) -> list[Violation]:
    violations: list[Violation] = []
    if row["task"] == "classification" and row["metric"] in {"mse", "rmse", "mae", "r2"}:
        violations.append(Violation("E201", "Regression metric used for classification task"))
    if row["task"] == "regression" and row["metric"] in {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }:
        violations.append(Violation("E202", "Classification metric used for regression task"))

    if row["preprocess_fit"] != "train":
        violations.append(Violation("E101", "Preprocessor is fitted outside the training partition"))
    if row["tune_on"] == "test":
        violations.append(
            Violation("E102", "Test partition is used for model selection or hyperparameter tuning")
        )
    if row["cv"] != "none" and row["cv_preprocess"] != "inside_fold":
        violations.append(
            Violation("E103", "Cross-validation preprocessing is fitted outside each fold")
        )
    if row["augment_on"] != "train":
        violations.append(Violation("E104", "Augmentation is applied outside the training partition"))

    if row["dataset_kind"] == "grouped" and row["group_split"] != "required":
        violations.append(Violation("E301", "Grouped observations may cross train/test partitions"))
    if row["dataset_kind"] == "temporal" and row["time_shuffle"] != "false":
        violations.append(Violation("E302", "Temporal order is violated by shuffling"))

    seed = str(row["seed"]).strip().lower()
    if seed in {"", "none", "nan", "null"}:
        violations.append(Violation("W401", "Random seed is not fixed"))
    return violations


def validate_dsl(text: str) -> list[Violation]:
    try:
        parsed = parse_dsl(text)
    except DSLParseError as error:
        return [Violation("E000", f"Invalid DSL syntax: {error}")]
    return validate_row(parsed)


def evaluate(frame: pd.DataFrame) -> dict[str, float | int]:
    true_labels = frame["gold_invalid"].astype(bool).to_numpy()
    if "dsl" in frame:
        predicted_labels = frame["dsl"].map(lambda text: bool(validate_dsl(text))).to_numpy()
    else:
        predicted_labels = frame.apply(lambda row: bool(validate_row(row)), axis=1).to_numpy()

    true_positive = int(((true_labels == 1) & (predicted_labels == 1)).sum())
    true_negative = int(((true_labels == 0) & (predicted_labels == 0)).sum())
    false_positive = int(((true_labels == 0) & (predicted_labels == 1)).sum())
    false_negative = int(((true_labels == 1) & (predicted_labels == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(frame)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "n": int(len(frame)),
    }
