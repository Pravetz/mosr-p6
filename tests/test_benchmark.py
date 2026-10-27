from __future__ import annotations

import unittest

from generate_benchmark import FEATURE_COLUMNS, VIOLATION_CATEGORIES, generate_benchmark
from verifier import evaluate, parse_dsl, validate_dsl


class BenchmarkTests(unittest.TestCase):
    def test_generated_benchmark_is_balanced_unique_and_grouped(self) -> None:
        frame = generate_benchmark(64)
        self.assertEqual(len(frame), 128)
        self.assertEqual(frame["base_id"].nunique(), 64)
        self.assertEqual(frame["gold_invalid"].value_counts().to_dict(), {False: 64, True: 64})
        self.assertFalse(frame.duplicated(list(FEATURE_COLUMNS)).any())
        self.assertEqual(
            set(frame.loc[frame["gold_invalid"], "category"]), set(VIOLATION_CATEGORIES)
        )

    def test_dsl_parser_and_all_rules_match_gold_labels(self) -> None:
        frame = generate_benchmark(64)
        first = parse_dsl(frame.iloc[0]["dsl"])
        self.assertEqual(first["id"], frame.iloc[0]["id"])
        self.assertEqual(first["task"], frame.iloc[0]["task"])
        result = evaluate(frame)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["tp"], 64)
        self.assertEqual(result["tn"], 64)

        expected_codes = {
            "global_fit_leakage": "E101",
            "test_tuning": "E102",
            "cv_preprocess_leakage": "E103",
            "test_augmentation": "E104",
            "metric_task_mismatch": "E201",
            "group_split_missing": "E301",
            "temporal_shuffle": "E302",
            "missing_seed": "W401",
        }
        for _, row in frame[frame["gold_invalid"]].iterrows():
            self.assertEqual(
                [violation.code for violation in validate_dsl(row["dsl"])],
                [expected_codes[row["category"]]],
            )

    def test_malformed_dsl_is_rejected(self) -> None:
        violations = validate_dsl("EXPERIMENT broken\nTASK classification")
        self.assertEqual([violation.code for violation in violations], ["E000"])


if __name__ == "__main__":
    unittest.main()
