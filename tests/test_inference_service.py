"""Regression tests for the model-testing (inference) service.

These cover the guards that run before PyTorch is ever touched: upload
validation, checkpoint discovery, and per-user rate limiting. The forward pass
itself needs torch and a real checkpoint, so it is exercised manually against a
completed run rather than here (see INFERENCE_FEATURE_REPORT.md).
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_service as inference


def _png_bytes(width: int = 8, height: int = 6) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_empty_upload_is_rejected(self):
        with self.assertRaises(inference.InferenceError) as caught:
            inference._open_verified_image(b"")
        self.assertIn("empty", str(caught.exception).lower())

    def test_non_image_payload_is_rejected(self):
        with self.assertRaises(inference.InferenceError) as caught:
            inference._open_verified_image(b"definitely not an image")
        self.assertIn("not a readable image", str(caught.exception))

    def test_oversized_payload_is_rejected_before_decoding(self):
        original = inference.MAX_INFERENCE_IMAGE_BYTES
        inference.MAX_INFERENCE_IMAGE_BYTES = 16
        try:
            with self.assertRaises(inference.InferenceError) as caught:
                inference._open_verified_image(b"x" * 64)
            self.assertIn("larger than", str(caught.exception))
        finally:
            inference.MAX_INFERENCE_IMAGE_BYTES = original

    def test_valid_png_is_converted_to_rgb(self):
        image = inference._open_verified_image(_png_bytes())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (8, 6))

    def test_pixel_budget_is_enforced(self):
        original = inference.MAX_INFERENCE_IMAGE_PIXELS
        inference.MAX_INFERENCE_IMAGE_PIXELS = 10
        try:
            with self.assertRaises(inference.InferenceError) as caught:
                inference._open_verified_image(_png_bytes(20, 20))
            self.assertIn("pixel limit", str(caught.exception))
        finally:
            inference.MAX_INFERENCE_IMAGE_PIXELS = original


class CheckpointDiscoveryTests(unittest.TestCase):
    def test_missing_checkpoint_reports_a_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(inference.InferenceError) as caught:
                inference._checkpoint_path(Path(temp))
            self.assertIn("best.pt", str(caught.exception))

    def test_best_checkpoint_is_preferred_over_last(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "last.pt").write_bytes(b"last")
            (run / "best.pt").write_bytes(b"best")
            self.assertEqual(inference._checkpoint_path(run).name, "best.pt")

    def test_last_checkpoint_is_used_when_best_is_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "last.pt").write_bytes(b"last")
            self.assertEqual(inference._checkpoint_path(run).name, "last.pt")


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        inference._rate_hits.clear()
        self._limit = inference.RATE_LIMIT_REQUESTS

    def tearDown(self):
        inference.RATE_LIMIT_REQUESTS = self._limit
        inference._rate_hits.clear()

    def test_requests_within_the_limit_are_allowed(self):
        inference.RATE_LIMIT_REQUESTS = 3
        for _ in range(3):
            inference.check_rate_limit("user-a")

    def test_exceeding_the_limit_raises(self):
        inference.RATE_LIMIT_REQUESTS = 2
        inference.check_rate_limit("user-a")
        inference.check_rate_limit("user-a")
        with self.assertRaises(inference.RateLimitExceeded):
            inference.check_rate_limit("user-a")

    def test_limits_are_tracked_per_user(self):
        inference.RATE_LIMIT_REQUESTS = 1
        inference.check_rate_limit("user-a")
        inference.check_rate_limit("user-b")  # must not inherit user-a's usage
        with self.assertRaises(inference.RateLimitExceeded):
            inference.check_rate_limit("user-a")


if __name__ == "__main__":
    unittest.main()
