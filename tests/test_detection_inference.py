"""Tests for detection inference parsing and helpers.

The YOLO path is tested with a fake Ultralytics model so it needs neither the
ultralytics package nor a real checkpoint. The Faster R-CNN path needs torch for
tensor handling and is skipped where torch is unavailable.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detection_inference as di
from inference_service import InferenceError

try:  # torch is present in the built API image, not necessarily on a dev host.
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def _tiny_image():
    from PIL import Image
    return Image.new("RGB", (100, 80), (128, 128, 128))


class _FakeArr:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _FakeArr(xyxy)
        self.conf = _FakeArr(conf)
        self.cls = _FakeArr(cls)
        self._n = len(conf)

    def __len__(self):
        return self._n


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeYolo:
    def __init__(self, result):
        self._result = result

    def predict(self, source, conf, verbose):  # noqa: ARG002 - mirrors ultralytics
        return [self._result]


class ClampThresholdTests(unittest.TestCase):
    def test_default_when_missing_or_bad(self):
        self.assertEqual(di._clamp_threshold(None), di.DEFAULT_SCORE_THRESHOLD)
        self.assertEqual(di._clamp_threshold("nope"), di.DEFAULT_SCORE_THRESHOLD)

    def test_bounds(self):
        self.assertEqual(di._clamp_threshold(-0.5), 0.0)
        self.assertEqual(di._clamp_threshold(5), 1.0)
        self.assertAlmostEqual(di._clamp_threshold(0.4), 0.4)


class CheckpointPathTests(unittest.TestCase):
    def test_yolo_prefers_weights_best(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "weights").mkdir()
            (run / "weights" / "best.pt").write_bytes(b"x")
            (run / "last.pt").write_bytes(b"x")
            self.assertEqual(di._yolo_checkpoint_path(run).name, "best.pt")
            self.assertEqual(di._yolo_checkpoint_path(run).parent.name, "weights")

    def test_yolo_missing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InferenceError):
                di._yolo_checkpoint_path(Path(d))

    def test_rcnn_prefers_best_over_last(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "best.pt").write_bytes(b"x")
            (run / "last.pt").write_bytes(b"x")
            self.assertEqual(di._rcnn_checkpoint_path(run).name, "best.pt")


class YoloParseTests(unittest.TestCase):
    def test_maps_names_and_boxes(self):
        boxes = _FakeBoxes(xyxy=[[10, 20, 30, 40]], conf=[0.9123], cls=[1])
        entry = {
            "kind": "yolo",
            "model": _FakeYolo(_FakeResult(boxes, {0: "cat", 1: "dog"})),
            "classes": ["cat", "dog"],
        }
        out = di._predict_yolo(entry, _tiny_image(), 0.25)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["className"], "dog")
        self.assertEqual(out[0]["box"], [10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(out[0]["percent"], 91.23, places=2)

    def test_empty_boxes(self):
        entry = {
            "kind": "yolo",
            "model": _FakeYolo(_FakeResult(_FakeBoxes([], [], []), {0: "cat"})),
            "classes": ["cat"],
        }
        self.assertEqual(di._predict_yolo(entry, _tiny_image(), 0.25), [])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class FasterRcnnParseTests(unittest.TestCase):
    def test_threshold_filters_and_labels_are_one_indexed(self):
        import torch

        class _FakeFrcnn:
            def __call__(self, images):  # noqa: ARG002
                return [{
                    "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                    "labels": torch.tensor([1, 2]),
                    "scores": torch.tensor([0.9, 0.1]),
                }]

        entry = {"kind": "faster_rcnn", "device": "cpu", "classes": ["cat", "dog"], "model": _FakeFrcnn()}
        out = di._predict_faster_rcnn(entry, _tiny_image(), 0.5)
        self.assertEqual(len(out), 1)  # 0.1 dropped by threshold
        self.assertEqual(out[0]["className"], "cat")  # label 1 -> classes[0]
        self.assertEqual(out[0]["box"], [1.0, 2.0, 3.0, 4.0])


class PredictDispatchTests(unittest.TestCase):
    def test_unknown_model_type_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InferenceError):
                di.load_detector(Path(d), "segformer")


if __name__ == "__main__":
    unittest.main()
