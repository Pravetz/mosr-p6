from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, shapiro, studentized_range, wilcoxon


ALGORITHMS = ["Baseline (local)", "Full validator", "Ablation - provenance"]


def cohen_dz(a: pd.Series, b: pd.Series) -> float:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(difference.mean() / difference.std(ddof=1))


def analyze(data_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    latency = pd.read_csv(data_dir / "pr5_latency_30runs.csv")
    raw_algorithms = pd.read_csv(data_dir / "pr5_three_algorithms_5datasets.csv")
    classification = pd.read_csv(data_dir / "pr5_classification_metrics.csv")

    baseline = latency["Baseline (local)"]
    full = latency["Full validator"]
    latency_result = {
        "n_paired_runs": int(len(latency)),
        "baseline_shapiro": dict(zip(("statistic", "pvalue"), map(float, shapiro(baseline)))),
        "full_shapiro": dict(zip(("statistic", "pvalue"), map(float, shapiro(full)))),
        "difference_shapiro": dict(zip(("statistic", "pvalue"), map(float, shapiro(full - baseline)))),
        "wilcoxon": dict(zip(("statistic", "pvalue"), map(float, wilcoxon(full, baseline)))),
        "baseline_median_us_per_spec": float(baseline.median()),
        "full_median_us_per_spec": float(full.median()),
    }

    variation = raw_algorithms.groupby(["dataset", "algorithm"])[["f1", "accuracy"]].nunique()
    repeated_values_are_identical = bool((variation == 1).all().all())

    # The ten rows per dataset/algorithm are exact copies, not independent observations.
    # Collapse them before inferential tests so the experimental unit is the dataset.
    independent = (
        raw_algorithms.groupby(["dataset", "algorithm"], as_index=False)[["f1", "accuracy"]]
        .mean()
    )
    pivot = independent.pivot(index="dataset", columns="algorithm", values="f1")
    friedman = friedmanchisquare(*(pivot[name] for name in ALGORITHMS))
    ranks = np.vstack(
        [rankdata(-row[ALGORITHMS].to_numpy(), method="average") for _, row in pivot.iterrows()]
    )
    average_ranks = pd.DataFrame({"algorithm": ALGORITHMS, "average_rank": ranks.mean(axis=0)})

    n_blocks = len(pivot)
    algorithm_count = len(ALGORITHMS)
    standard_error = math.sqrt(algorithm_count * (algorithm_count + 1) / (6 * n_blocks))
    comparisons = []
    for i in range(algorithm_count):
        for j in range(i + 1, algorithm_count):
            rank_difference = abs(average_ranks.average_rank.iloc[i] - average_ranks.average_rank.iloc[j])
            q = rank_difference / standard_error
            comparisons.append(
                {
                    "algorithm_1": ALGORITHMS[i],
                    "algorithm_2": ALGORITHMS[j],
                    "rank_difference": float(rank_difference),
                    "p_nemenyi": float(
                        studentized_range.sf(q * math.sqrt(2), algorithm_count, np.inf)
                    ),
                    "cohen_dz_across_datasets": cohen_dz(
                        pivot[ALGORITHMS[i]], pivot[ALGORITHMS[j]]
                    ),
                }
            )
    posthoc = pd.DataFrame(comparisons)

    result = {
        "analysis_unit": "dataset",
        "effective_independent_blocks": n_blocks,
        "raw_rows": int(len(raw_algorithms)),
        "nominal_repeats_per_dataset": int(raw_algorithms["run"].nunique()),
        "repeated_values_are_identical": repeated_values_are_identical,
        "latency": latency_result,
        "friedman": {"statistic": float(friedman.statistic), "pvalue": float(friedman.pvalue)},
        "classification_metrics": classification.to_dict(orient="records"),
    }
    return result, average_ranks, posthoc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/pr5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pr5"))
    args = parser.parse_args()

    result, ranks, posthoc = analyze(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ranks.to_csv(args.output_dir / "average_ranks_corrected.csv", index=False)
    posthoc.to_csv(args.output_dir / "posthoc_corrected.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nCorrected post-hoc comparisons (five independent dataset blocks):")
    print(posthoc.to_string(index=False))


if __name__ == "__main__":
    main()
