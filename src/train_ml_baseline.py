from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from provenance import capture_git_provenance


FEATURE_COLUMNS = [
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
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/benchmark_specs.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/ml_baseline_metrics.json")
    parser.add_argument("--use-mlflow", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    frame = pd.read_csv(data_path)
    if "base_id" not in frame:
        raise ValueError(
            "The benchmark must contain base_id groups. The legacy PR4 dataset is unsuitable "
            "for a leakage-free train/test evaluation."
        )
    if frame.duplicated(FEATURE_COLUMNS).any():
        raise ValueError("Duplicate feature vectors detected; refusing a leaked row-wise evaluation")

    features = frame[FEATURE_COLUMNS].astype(str)
    labels = frame["gold_invalid"].astype(int)
    groups = frame["base_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_index, test_index = next(splitter.split(features, labels, groups))
    x_train, x_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Group leakage detected: {sorted(overlap)}")

    preprocessing = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS)]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=args.seed)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    provenance = capture_git_provenance()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "seed": args.seed,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": 0,
        "data_sha256": data_sha256,
        **provenance,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.use_mlflow:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        mlflow.set_experiment("mlspec-ml-baseline")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "model": "LogisticRegression",
                    "test_size": 0.30,
                    "split_strategy": "GroupShuffleSplit(base_id)",
                    "data_sha256": data_sha256,
                    "git_commit": provenance["git_commit"],
                    "git_dirty": provenance["git_dirty"],
                }
            )
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                mlflow.log_metric(key, metrics[key])
            mlflow.log_artifact(str(output))
            mlflow.sklearn.log_model(model, artifact_path="model")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
