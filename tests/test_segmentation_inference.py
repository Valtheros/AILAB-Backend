"""Tests for segmentation inference helpers and output parsing.

Semantic/instance forward passes need torch (+PIL/numpy) and are skipped where
unavailable; the pure helpers always run.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import segmentation_inference as si
from inference_service import InferenceError

try:
    import torch  # noqa: F401
    from PIL import Image  # noqa: F401
    import numpy  # noqa: F401
    HAS_STACK = True
except Exception:  # pragma: no cover
    HAS_STACK = False


def _tiny_image():
    from PIL import Image
    return Image.new("RGB", (40, 30), (100, 120, 140))


class HelperTests(unittest.TestCase):
    def test_hex(self):
        self.assertEqual(si._hex((239, 68, 68)), "#ef4444")

    def test_semantic_names_prepend_background_and_pad(self):
        config = {"dataset_metadata": {"classes": ["road"]}}
        self.assertEqual(si._semantic_class_names(config, 2), ["background", "road"])
        # Pads when fewer names than classes.
        self.assertEqual(si._semantic_class_names({}, 3), ["background", "class 1", "class 2"])

    def test_unknown_model_type_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InferenceError):
                si.load_segmenter(Path(d), "unet")

    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InferenceError):
                si._checkpoint_path(Path(d))


@unittest.skipUnless(HAS_STACK, "torch/PIL/numpy not installed")
class SemanticParseTests(unittest.TestCase):
    def test_overlay_and_legend(self):
        import torch

        # Logits that make the top half class 1 and the bottom half class 0.
        size = 16
        logits = torch.zeros(1, 2, size, size)
        logits[0, 1, : size // 2, :] = 10.0  # class 1 wins in the top half
        logits[0, 0, size // 2 :, :] = 10.0  # class 0 wins in the bottom half

        class _FakeModel:
            def __call__(self, _tensor):
                return logits

        entry = {
            "kind": "semantic", "model": _FakeModel(), "num_classes": 2,
            "image_size": size, "device": "cpu", "classes": ["background", "road"],
        }
        out = si._predict_semantic(entry, _tiny_image(), 0.5)
        self.assertEqual(out["segmentation"]["kind"], "semantic")
        self.assertTrue(out["segmentation"]["overlay"].startswith("data:image/png;base64,"))
        legend = out["segmentation"]["legend"]
        self.assertEqual([e["name"] for e in legend], ["road"])  # background excluded
        self.assertAlmostEqual(legend[0]["percent"], 50.0, delta=2.0)


@unittest.skipUnless(HAS_STACK, "torch/PIL/numpy not installed")
class InstanceParseTests(unittest.TestCase):
    def test_threshold_and_mask_overlay(self):
        import torch

        w, h = 40, 30
        mask_keep = torch.zeros(1, h, w)
        mask_keep[0, 0:10, 0:10] = 1.0
        mask_drop = torch.ones(1, h, w)

        class _FakeModel:
            def __call__(self, _images):
                return [{
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 5.0, 5.0]]),
                    "labels": torch.tensor([1, 2]),
                    "scores": torch.tensor([0.9, 0.2]),
                    "masks": torch.stack([mask_keep, mask_drop]),  # (2,1,H,W)
                }]

        entry = {"kind": "instance", "model": _FakeModel(), "device": "cpu", "classes": ["fire", "smoke"]}
        out = si._predict_instance(entry, _tiny_image(), 0.5)
        self.assertEqual(out["count"], 1)  # second instance dropped by threshold
        seg = out["segmentation"]
        self.assertEqual(seg["kind"], "instance")
        self.assertEqual(seg["instances"][0]["className"], "fire")  # label 1 -> classes[0]
        self.assertTrue(seg["overlay"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
