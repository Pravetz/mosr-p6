from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd


VIOLATION_CATEGORIES = (
    "global_fit_leakage",
    "test_tuning",
    "cv_preprocess_leakage",
    "test_augmentation",
    "group_split_missing",
    "temporal_shuffle",
    "metric_task_mismatch",
    "missing_seed",
)

FEATURE_COLUMNS = (
    "task",
    "model_family",
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
)


def _valid_base(index: int) -> dict[str, object]:
    task = "classification" if index % 2 == 0 else "regression"
    dataset_kind = ("iid", "grouped", "temporal")[index % 3]
    if dataset_kind == "grouped":
        split = "group"
    elif dataset_kind == "temporal":
        split = "temporal"
    else:
        split = "stratified" if task == "classification" else "random"

    class_metrics = ("accuracy", "precision", "recall", "f1", "roc_auc")
    regression_metrics = ("mse", "rmse", "mae", "r2")
    metric_pool = class_metrics if task == "classification" else regression_metrics
    classification_models = (
        "logistic_regression",
        "random_forest",
        "linear_svm",
        "gradient_boosting",
        "naive_bayes",
        "knn",
        "decision_tree",
        "extra_trees",
    )
    regression_models = (
        "ridge",
        "random_forest_regressor",
        "linear_svr",
        "gradient_boosting_regressor",
        "elastic_net",
        "knn_regressor",
        "decision_tree_regressor",
        "extra_trees_regressor",
    )
    model_pool = classification_models if task == "classification" else regression_models
    cv = ("none", "5fold", "10fold")[index % 3]

    return {
        "task": task,
        "model_family": model_pool[(index // len(VIOLATION_CATEGORIES)) % len(model_pool)],
        "split": split,
        "seed": 1000 + index,
        "preprocess_fit": "train",
        "preprocess_apply": "train,validation,test",
        "tune_on": "validation",
        "metric": metric_pool[index % len(metric_pool)],
        "cv": cv,
        "cv_preprocess": "inside_fold",
        "augment_on": "train",
        "dataset_kind": dataset_kind,
        "group_split": "required" if dataset_kind == "grouped" else "not_required",
        "time_shuffle": "false" if dataset_kind == "temporal" else "not_applicable",
    }


def _inject_violation(config: dict[str, object], category: str) -> None:
    if category == "global_fit_leakage":
        config["preprocess_fit"] = "all"
    elif category == "test_tuning":
        config["tune_on"] = "test"
    elif category == "cv_preprocess_leakage":
        config["cv"] = "5fold"
        config["cv_preprocess"] = "outside_fold"
    elif category == "test_augmentation":
        config["augment_on"] = "train,test"
    elif category == "group_split_missing":
        config.update(
            dataset_kind="grouped",
            split="group",
            group_split="false",
            time_shuffle="not_applicable",
        )
    elif category == "temporal_shuffle":
        config.update(
            dataset_kind="temporal",
            split="temporal",
            group_split="not_required",
            time_shuffle="true",
        )
    elif category == "metric_task_mismatch":
        config["metric"] = "rmse" if config["task"] == "classification" else "f1"
    elif category == "missing_seed":
        config["seed"] = "none"
    else:
        raise ValueError(f"Unknown violation category: {category}")


def _to_dsl(experiment_id: str, config: dict[str, object]) -> str:
    lines = [f"EXPERIMENT {experiment_id}"]
    lines.extend(f"{key.upper()} {config[key]}" for key in FEATURE_COLUMNS)
    return "\n".join(lines)


def generate_benchmark(base_count: int = 64) -> pd.DataFrame:
    """Generate paired valid/invalid specifications without duplicate feature vectors."""
    if base_count < len(VIOLATION_CATEGORIES) * 2:
        raise ValueError("base_count must be at least 16 for useful category coverage")

    rows: list[dict[str, object]] = []
    for index in range(base_count):
        base_id = f"base_{index:03d}"
        valid = _valid_base(index)
        valid_id = f"{base_id}_valid"
        rows.append(
            {
                "id": valid_id,
                "base_id": base_id,
                **valid,
                "category": "valid",
                "gold_invalid": False,
                "dsl": _to_dsl(valid_id, valid),
            }
        )

        category = VIOLATION_CATEGORIES[index % len(VIOLATION_CATEGORIES)]
        invalid = deepcopy(valid)
        _inject_violation(invalid, category)
        invalid_id = f"{base_id}_{category}"
        rows.append(
            {
                "id": invalid_id,
                "base_id": base_id,
                **invalid,
                "category": category,
                "gold_invalid": True,
                "dsl": _to_dsl(invalid_id, invalid),
            }
        )

    result = pd.DataFrame(rows)
    duplicated = result.duplicated(list(FEATURE_COLUMNS), keep=False)
    if duplicated.any():
        duplicate_ids = result.loc[duplicated, "id"].tolist()
        raise RuntimeError(f"Generator produced duplicate feature vectors: {duplicate_ids}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/benchmark_specs.csv")
    parser.add_argument("--base-count", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark = generate_benchmark(args.base_count)
    benchmark.to_csv(output, index=False)
    print(
        f"Wrote {len(benchmark)} unique specifications "
        f"({benchmark['gold_invalid'].value_counts().to_dict()}) to {output}"
    )


if __name__ == "__main__":
    main()
