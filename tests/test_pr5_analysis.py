from __future__ import annotations

import unittest
from pathlib import Path

from analyze_pr5 import analyze


class PR5AnalysisTests(unittest.TestCase):
    def test_pr5_analysis_uses_independent_dataset_blocks(self) -> None:
        result, ranks, posthoc = analyze(Path("data/pr5"))
        self.assertEqual(result["raw_rows"], 150)
        self.assertEqual(result["effective_independent_blocks"], 5)
        self.assertTrue(result["repeated_values_are_identical"])
        self.assertAlmostEqual(result["friedman"]["pvalue"], 0.018315638888734116)
        self.assertEqual(len(ranks), 3)
        ablation = posthoc[posthoc["algorithm_2"] == "Ablation - provenance"]
        self.assertTrue((ablation["p_nemenyi"] > 0.05).all())


if __name__ == "__main__":
    unittest.main()
