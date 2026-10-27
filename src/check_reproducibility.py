from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DETERMINISTIC_KEYS = (
    "seed",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "n",
    "data_sha256",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.runs < 2:
        raise ValueError("At least two runs are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run_number in range(1, args.runs + 1):
        output = args.output_dir / f"run{run_number}.json"
        subprocess.run(
            [
                sys.executable,
                "src/run_experiment.py",
                "--seed",
                str(args.seed),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(output.read_text(encoding="utf-8")))

    reference = {key: results[0][key] for key in DETERMINISTIC_KEYS}
    for run_number, result in enumerate(results[1:], start=2):
        candidate = {key: result[key] for key in DETERMINISTIC_KEYS}
        if candidate != reference:
            raise RuntimeError(f"Run {run_number} differs in deterministic outputs")

    fields = [
        "seed",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "elapsed_s",
        "data_sha256",
        "git_commit",
        "git_dirty",
    ]
    with (args.output_dir / "reproducibility_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: result[key] for key in fields} for result in results)
    print(f"Verified {args.runs} reproducible runs with seed {args.seed}")


if __name__ == "__main__":
    main()
