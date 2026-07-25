"""Regression tests for run comparison metric resolution and series building.

The endpoint itself is exercised against real runs over HTTP (see
COMPARE_RUNS_FEATURE_REPORT.md). These tests pin the pure logic: which metric
gets picked for each task type, and how best/final points are derived.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_comparison import build_run_comparison, resolve_metric, summarise_comparison


CLASSIFICATION_ROWS = [
    {"epoch": "1", "train/loss": "1.08", "train/accuracy": "0.40", "val/loss": "1.07", "val/accuracy": "0.39"},
    {"epoch": "2", "train/loss": "0.90", "train/accuracy": "0.66", "val/loss": "1.00", "val/accuracy": "0.57"},
    {"epoch": "3", "train/loss": "0.67", "train/accuracy": "0.89", "val/loss": "0.83", "val/accuracy": "0.73"},
]

YOLO_ROWS = [
    {"epoch": "1", "train/box_loss": "1.2", "metrics/mAP50(B)": "0.51", "val/box_loss": "1.1"},
    {"epoch": "2", "train/box_loss": "0.9", "metrics/mAP50(B)": "0.88", "val/box_loss": "0.8"},
    {"epoch": "3", "train/box_loss": "0.6", "metrics/mAP50(B)": "0.72", "val/box_loss": "0.7"},
]

DEEPLAB_ROWS = [
    {"epoch": "1", "train/loss": "0.9", "train/pixel_accuracy": "0.80", "val/loss": "0.8", "val/pixel_accuracy": "0.82"},
    {"epoch": "2", "train/loss": "0.4", "train/pixel_accuracy": "0.95", "val/loss": "0.5", "val/pixel_accuracy": "0.98"},
]

MASKRCNN_ROWS = [
    {"epoch": "1", "train/loss": "0.50", "val/loss": "0.40"},
    {"epoch": "2", "train/loss": "0.10", "val/loss": "0.20"},
]


def _task(**overrides):
    row = {
        "id": "11111111-1111-1111-1111-111111111111",
        "run_slug": "demo_run",
        "display_name": "demo",
        "task_type": "image_classification",
        "model_type": "resnet",
        "model_name": "resnet50",
        "dataset_slug": "demo.folder",
        "params": {"epochs": 3, "batch_size": 32},
    }
    row.update(overrides)
    return row


class MetricResolutionTests(unittest.TestCase):
    def test_classification_prefers_accuracy(self):
        metric = resolve_metric("image_classification", set(CLASSIFICATION_ROWS[0]))
        self.assertEqual(metric["key"], "accuracy")
        self.assertEqual(metric["direction"], "higher")

    def test_detection_uses_map50(self):
        metric = resolve_metric("object_detection", set(YOLO_ROWS[0]))
        self.assertEqual(metric["key"], "map50")
        # Ultralytics records mAP on validation only.
        self.assertIsNone(metric["trainColumn"])

    def test_detection_without_map_falls_back_to_loss(self):
        # Faster R-CNN records plain train/val loss and no mAP column.
        metric = resolve_metric("object_detection", {"epoch", "train/loss", "val/loss"})
        self.assertEqual(metric["key"], "loss")
        self.assertEqual(metric["direction"], "lower")

    def test_segmentation_prefers_pixel_accuracy_when_recorded(self):
        metric = resolve_metric("segmentation", set(DEEPLAB_ROWS[0]))
        self.assertEqual(metric["key"], "pixel_accuracy")
        self.assertEqual(metric["direction"], "higher")

    def test_segmentation_falls_back_to_loss(self):
        metric = resolve_metric("segmentation", set(MASKRCNN_ROWS[0]))
        self.assertEqual(metric["key"], "loss")
        self.assertEqual(metric["direction"], "lower")

    def test_unknown_columns_resolve_to_nothing(self):
        self.assertIsNone(resolve_metric("image_classification", {"epoch", "time"}))


class SeriesTests(unittest.TestCase):
    def test_classification_series_and_best_point(self):
        result = build_run_comparison(
            task_row=_task(),
            metric_rows=CLASSIFICATION_ROWS,
            config={"extra_args": {"architecture": "resnet50", "learning_rate": 0.001}},
        )
        self.assertEqual(result["series"]["epoch"], [1.0, 2.0, 3.0])
        self.assertEqual(result["series"]["valMetric"], [0.39, 0.57, 0.73])
        self.assertEqual(result["series"]["trainMetric"], [0.40, 0.66, 0.89])
        # Highest validation accuracy, not the last one.
        self.assertEqual(result["bestMetric"], {"value": 0.73, "epoch": 3.0})
        self.assertEqual(result["finalMetric"], {"value": 0.73})
        self.assertEqual(result["architecture"], "resnet50")
        self.assertEqual(result["trainingParams"]["learningRate"], 0.001)

    def test_best_point_picks_peak_not_final(self):
        result = build_run_comparison(
            task_row=_task(task_type="object_detection", model_type="yolo"),
            metric_rows=YOLO_ROWS,
            config={},
        )
        self.assertEqual(result["bestMetric"], {"value": 0.88, "epoch": 2.0})
        self.assertEqual(result["finalMetric"], {"value": 0.72})
        # No training counterpart exists for mAP.
        self.assertIsNone(result["series"]["trainMetric"])

    def test_lower_is_better_metric_takes_the_minimum(self):
        result = build_run_comparison(
            task_row=_task(task_type="segmentation", model_type="mask_rcnn"),
            metric_rows=MASKRCNN_ROWS,
            config={},
        )
        self.assertEqual(result["metric"]["direction"], "lower")
        self.assertEqual(result["bestMetric"], {"value": 0.20, "epoch": 2.0})

    def test_blank_cells_do_not_break_the_series(self):
        rows = [
            {"epoch": "1", "train/accuracy": "0.5", "val/accuracy": ""},
            {"epoch": "2", "train/accuracy": "0.7", "val/accuracy": "0.6"},
        ]
        result = build_run_comparison(task_row=_task(), metric_rows=rows, config={})
        self.assertEqual(result["series"]["valMetric"], [None, 0.6])
        self.assertEqual(result["bestMetric"], {"value": 0.6, "epoch": 2.0})

    def test_run_without_metrics_yields_no_metric(self):
        result = build_run_comparison(task_row=_task(), metric_rows=[], config={})
        self.assertIsNone(result["metric"])
        self.assertIsNone(result["bestMetric"])
        self.assertEqual(result["epochsRecorded"], 0)


class SummaryTests(unittest.TestCase):
    def _run(self, task_type, rows):
        return build_run_comparison(task_row=_task(task_type=task_type), metric_rows=rows, config={})

    def test_matching_metrics_report_a_shared_metric(self):
        runs = [
            self._run("image_classification", CLASSIFICATION_ROWS),
            self._run("image_classification", CLASSIFICATION_ROWS),
        ]
        summary = summarise_comparison(runs)
        self.assertEqual(summary["sharedMetric"]["key"], "accuracy")
        self.assertEqual(summary["warnings"], [])

    def test_mixed_metrics_are_flagged_instead_of_silently_charted(self):
        runs = [
            self._run("segmentation", DEEPLAB_ROWS),   # pixel accuracy
            self._run("segmentation", MASKRCNN_ROWS),  # loss
        ]
        summary = summarise_comparison(runs)
        self.assertIsNone(summary["sharedMetric"])
        self.assertTrue(any("different metrics" in warning for warning in summary["warnings"]))

    def test_mixed_task_types_are_flagged(self):
        runs = [
            self._run("image_classification", CLASSIFICATION_ROWS),
            self._run("object_detection", YOLO_ROWS),
        ]
        summary = summarise_comparison(runs)
        self.assertTrue(any("task types" in warning for warning in summary["warnings"]))


if __name__ == "__main__":
    unittest.main()
