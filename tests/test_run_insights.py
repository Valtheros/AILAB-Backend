"""Rule-boundary tests for run_insights.

Real runs on the demo dataset all overfit hard, so these synthetic rows pin the
mild / none / stable / test-gap branches that live data never reaches. Numbers
are chosen so the expected code is unambiguous.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_insights import analyse_run, compare_insights
from run_comparison import build_run_comparison


def _codes(result):
    return {item["code"]: item for item in result["insights"]}


def _cls_rows(pairs):
    """pairs: list of (train_acc, val_acc); epoch auto-numbered from 1."""
    return [
        {"epoch": str(i + 1), "train/accuracy": f"{ta}", "val/accuracy": f"{va}",
         "train/loss": "0.5", "val/loss": "0.6"}
        for i, (ta, va) in enumerate(pairs)
    ]


class OverfittingBandTests(unittest.TestCase):
    def test_high_overfitting(self):
        rows = _cls_rows([(0.6, 0.55), (0.99, 0.70)])  # gap 29pp
        codes = _codes(analyse_run(rows, "image_classification"))
        self.assertIn("overfitting.high", codes)
        self.assertEqual(codes["overfitting.high"]["level"], "danger")

    def test_mild_overfitting(self):
        rows = _cls_rows([(0.6, 0.55), (0.80, 0.70)])  # gap 10pp -> mild (8..15)
        codes = _codes(analyse_run(rows, "image_classification"))
        self.assertIn("overfitting.mild", codes)
        self.assertAlmostEqual(codes["overfitting.mild"]["params"]["gap"], 10.0, places=1)

    def test_no_overfitting(self):
        rows = _cls_rows([(0.6, 0.58), (0.75, 0.72)])  # gap 3pp -> none
        codes = _codes(analyse_run(rows, "image_classification"))
        self.assertIn("overfitting.none", codes)
        self.assertEqual(codes["overfitting.none"]["level"], "success")


class BestEpochTests(unittest.TestCase):
    def test_best_epoch_is_the_peak_validation(self):
        rows = _cls_rows([(0.5, 0.60), (0.7, 0.82), (0.9, 0.78)])
        codes = _codes(analyse_run(rows, "image_classification"))
        self.assertEqual(codes["best_epoch_acc"]["params"]["epoch"], 2)
        self.assertAlmostEqual(codes["best_epoch_acc"]["params"]["value"], 82.0, places=1)


class StabilityTests(unittest.TestCase):
    def test_unstable_validation_flagged(self):
        # last-5 val swings widely.
        pairs = [(0.9, 0.50), (0.9, 0.75), (0.9, 0.55), (0.9, 0.78), (0.9, 0.52)]
        codes = _codes(analyse_run(_cls_rows(pairs), "image_classification"))
        self.assertIn("val_unstable", codes)

    def test_stable_validation_not_flagged(self):
        pairs = [(0.9, 0.70), (0.9, 0.71), (0.9, 0.70), (0.9, 0.72), (0.9, 0.71)]
        codes = _codes(analyse_run(_cls_rows(pairs), "image_classification"))
        self.assertNotIn("val_unstable", codes)


class TestGapTests(unittest.TestCase):
    def test_large_test_gap_is_flagged(self):
        rows = _cls_rows([(0.8, 0.60), (0.9, 0.75)])  # best val 0.75
        codes = _codes(analyse_run(rows, "image_classification", {"test_accuracy": 0.60}))  # gap 15pp
        self.assertIn("test_gap.high", codes)

    def test_test_gap_carries_both_real_numbers(self):
        # best val 0.82 at epoch 2 (not the final epoch's 0.78); test 0.65.
        rows = _cls_rows([(0.7, 0.70), (0.9, 0.82), (0.95, 0.78)])
        params = _codes(analyse_run(rows, "image_classification", {"test_accuracy": 0.65}))["test_gap.high"]["params"]
        self.assertAlmostEqual(params["best_val"], 82.0, places=1)
        self.assertEqual(params["best_epoch"], 2)
        self.assertAlmostEqual(params["test"], 65.0, places=1)
        self.assertAlmostEqual(params["gap"], 17.0, places=1)

    def test_close_test_gap_reports_generalisation(self):
        rows = _cls_rows([(0.8, 0.60), (0.9, 0.75)])
        codes = _codes(analyse_run(rows, "image_classification", {"test_accuracy": 0.72}))  # gap 3pp
        self.assertIn("test_gap.ok", codes)
        self.assertEqual(codes["test_gap.ok"]["level"], "success")

    def test_no_test_eval_means_no_test_insight(self):
        codes = _codes(analyse_run(_cls_rows([(0.8, 0.6), (0.9, 0.75)]), "image_classification", None))
        self.assertNotIn("test_gap.high", codes)
        self.assertNotIn("test_gap.ok", codes)


class EdgeTests(unittest.TestCase):
    def test_empty_rows_yield_no_insights(self):
        result = analyse_run([], "image_classification")
        self.assertEqual(result["insights"], [])

    def test_detection_map_val_only_skips_overfitting(self):
        # YOLO records mAP on validation only -> no train counterpart, so the
        # overfitting rule must not fire (and must not crash).
        rows = [{"epoch": "1", "metrics/mAP50(B)": "0.5", "train/box_loss": "1.0", "val/box_loss": "1.1"},
                {"epoch": "2", "metrics/mAP50(B)": "0.9", "train/box_loss": "0.6", "val/box_loss": "0.7"}]
        codes = _codes(analyse_run(rows, "object_detection"))
        self.assertNotIn("overfitting.high", codes)
        self.assertNotIn("overfitting.mild", codes)
        self.assertIn("best_epoch_acc", codes)  # mAP is higher-is-better


class CompareInsightTests(unittest.TestCase):
    def _run(self, name, pairs, epochs_recorded=None):
        rows = _cls_rows(pairs)
        row = {"id": name, "run_slug": name, "display_name": name,
               "task_type": "image_classification", "model_type": "resnet",
               "model_name": "resnet50", "dataset_slug": "d", "params": {}}
        run = build_run_comparison(task_row=row, metric_rows=rows, config={})
        return run

    def test_flags_overfit_extremes(self):
        # run A overfits hard; run B cleaner.
        a = self._run("A", [(0.7, 0.65), (0.99, 0.70)])
        b = self._run("B", [(0.6, 0.58), (0.75, 0.72)])
        codes = {i["code"]: i for i in compare_insights([a, b])}
        self.assertEqual(codes["compare.overfit_most"]["params"]["run"], "A")
        self.assertEqual(codes["compare.overfit_least"]["params"]["run"], "B")
        self.assertIn("compare.best_value", codes)
        self.assertNotIn("compare.fastest", codes)  # removed: not useful in practice

    def test_single_run_has_no_comparison(self):
        a = self._run("A", [(0.7, 0.65), (0.9, 0.70)])
        self.assertEqual(compare_insights([a]), [])

    def test_least_overfit_still_above_threshold_warns(self):
        # both overfit hard: A gap 30pp, B gap 20pp. B is "least" but still >15.
        a = self._run("A", [(0.7, 0.65), (0.99, 0.69)])
        b = self._run("B", [(0.6, 0.58), (0.95, 0.75)])
        codes = {i["code"]: i for i in compare_insights([a, b])}
        self.assertIn("compare.overfit_least_high", codes)
        self.assertNotIn("compare.overfit_least", codes)
        self.assertEqual(codes["compare.overfit_least_high"]["params"]["run"], "B")
        self.assertEqual(codes["compare.overfit_least_high"]["level"], "warning")

    def test_least_overfit_below_threshold_stays_green(self):
        a = self._run("A", [(0.7, 0.65), (0.90, 0.70)])   # gap 20pp
        b = self._run("B", [(0.6, 0.58), (0.75, 0.72)])   # gap 3pp -> below 15
        codes = {i["code"]: i for i in compare_insights([a, b])}
        self.assertIn("compare.overfit_least", codes)
        self.assertEqual(codes["compare.overfit_least"]["level"], "success")


if __name__ == "__main__":
    unittest.main()
