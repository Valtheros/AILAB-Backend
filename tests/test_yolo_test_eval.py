"""Tests for the YOLO held-out test-evaluation helpers.

Only the pure helpers are covered here; the actual `model.val(split="test")`
call needs ultralytics and a trained checkpoint, so it is exercised manually
against real runs rather than in unit tests.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.trainers.yolo_trainer import (
    F1_COLUMN,
    _append_f1_column,
    _class_names,
    _count_images,
    _f1,
    _resolve_test_dir,
)


class ClassNamesTests(unittest.TestCase):
    def test_dict_is_ordered_by_index(self):
        self.assertEqual(_class_names({"names": {1: "b", 0: "a", 2: "c"}}), ["a", "b", "c"])

    def test_list_passthrough(self):
        self.assertEqual(_class_names({"names": ["cat", "dog"]}), ["cat", "dog"])

    def test_missing_names(self):
        self.assertEqual(_class_names({}), [])


class ResolveTestDirTests(unittest.TestCase):
    def test_none_when_no_test_key(self):
        self.assertIsNone(_resolve_test_dir("/x/data.yaml", {"train": "train/images"}))

    def test_resolves_against_path_key(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test" / "images").mkdir(parents=True)
            resolved = _resolve_test_dir(str(root / "data.yaml"), {"path": str(root), "test": "test/images"})
            self.assertEqual(resolved, (root / "test" / "images").resolve())

    def test_none_when_declared_test_dir_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_resolve_test_dir(str(Path(d) / "data.yaml"), {"path": d, "test": "test/images"}))

    def test_falls_back_to_yaml_dir_without_path_key(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test" / "images").mkdir(parents=True)
            resolved = _resolve_test_dir(str(root / "data.yaml"), {"test": "test/images"})
            self.assertEqual(resolved, (root / "test" / "images").resolve())


class CountImagesTests(unittest.TestCase):
    def test_counts_images_recursively(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.jpg").write_bytes(b"x")
            (root / "sub").mkdir()
            (root / "sub" / "b.png").write_bytes(b"x")
            (root / "notes.txt").write_bytes(b"x")  # ignored
            self.assertEqual(_count_images(root), 2)

    def test_counts_lines_of_txt_listing(self):
        with tempfile.TemporaryDirectory() as d:
            listing = Path(d) / "test.txt"
            listing.write_text("img1.jpg\nimg2.jpg\n\nimg3.jpg\n", encoding="utf-8")
            self.assertEqual(_count_images(listing), 3)


class F1Tests(unittest.TestCase):
    def test_harmonic_mean(self):
        self.assertAlmostEqual(_f1(0.6, 0.4), 0.48, places=6)
        self.assertAlmostEqual(_f1(1.0, 1.0), 1.0, places=6)

    def test_zero_precision_and_recall_does_not_divide_by_zero(self):
        self.assertEqual(_f1(0.0, 0.0), 0.0)

    def _write_csv(self, path: Path, rows: list[str]) -> None:
        header = "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),lr/pg0"
        path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    def test_appends_column_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "results.csv"
            self._write_csv(csv_path, ["1,0.6,0.4,0.5,0.01", "2,0,0,0,0.01"])
            self.assertEqual(_append_f1_column(csv_path), 2)
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn(F1_COLUMN, text.splitlines()[0])
            # Original columns survive.
            for column in ("epoch", "metrics/mAP50(B)", "lr/pg0"):
                self.assertIn(column, text.splitlines()[0])
            import csv as _csv
            rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
            self.assertAlmostEqual(float(rows[0][F1_COLUMN]), 0.48, places=4)
            self.assertAlmostEqual(float(rows[1][F1_COLUMN]), 0.0, places=6)

    def test_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "results.csv"
            self._write_csv(csv_path, ["1,0.6,0.4,0.5,0.01"])
            self.assertEqual(_append_f1_column(csv_path), 1)
            self.assertEqual(_append_f1_column(csv_path), 0)

    def test_skips_csv_without_precision_recall(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "results.csv"
            csv_path.write_text("epoch,train/loss\n1,0.5\n", encoding="utf-8")
            self.assertEqual(_append_f1_column(csv_path), 0)

    def test_missing_file_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_append_f1_column(Path(d) / "nope.csv"), 0)


if __name__ == "__main__":
    unittest.main()
