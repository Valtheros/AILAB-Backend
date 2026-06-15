from __future__ import annotations

import unittest
from unittest.mock import patch

from resource_guard import (
    ResourcePlanError,
    enforce_resource_plan,
    get_resource_profile,
    validate_resource_plan,
)


TEST_PROFILE = {
    "hardware": {
        "gpus": [{"index": "0", "name": "NVIDIA GeForce RTX 4070 Ti", "vram_total_mb": 12282, "vram_free_mb": 11000}],
        "gpu_available": True,
        "primary_gpu": {"index": "0", "name": "NVIDIA GeForce RTX 4070 Ti", "vram_total_mb": 12282, "vram_free_mb": 11000},
        "default_device": "0",
        "system_ram_total_gb": 16,
        "system_ram_available_gb": 12,
    },
    "safe_limits": {"gpu_vram_mb": 10500, "system_ram_gb": 12, "workers": 4},
    "policy": "auto_safe",
    "notes": [],
}


class ResourceGuardTests(unittest.TestCase):
    def test_deeplab_large_image_and_batch_rejected(self):
        with self.assertRaises(ResourcePlanError) as caught:
            enforce_resource_plan(
                "deeplabv3plus",
                {"image_size": 2048, "workers": 4, "device": "0"},
                batch_size=16,
                profile=TEST_PROFILE,
            )
        self.assertIn("DeepLabV3+ image_size 2048", str(caught.exception))
        self.assertIn("batch_size <=", str(caught.exception))

    def test_mask_rcnn_batch_too_high_rejected(self):
        plan = validate_resource_plan(
            "mask_rcnn",
            {"image_size": 1024, "max_size": 1333, "device": "0"},
            batch_size=8,
            profile=TEST_PROFILE,
        )
        self.assertFalse(plan["ok"])
        self.assertTrue(any("batch_size 8" in item for item in plan["errors"]))

    def test_faster_rcnn_batch_too_high_rejected(self):
        plan = validate_resource_plan(
            "faster_rcnn",
            {"image_size": 640, "max_size": 1333, "device": "0"},
            batch_size=8,
            profile=TEST_PROFILE,
        )
        self.assertFalse(plan["ok"])
        self.assertTrue(any("batch_size 8" in item for item in plan["errors"]))

    def test_yolo_n_640_safe_passes(self):
        plan = validate_resource_plan(
            "yolo",
            {"imgsz": 640, "model_size": "n", "workers": 4, "device": "0"},
            batch_size=16,
            profile=TEST_PROFILE,
        )
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["errors"], [])

    def test_yolo_large_imgsz_rejected_with_suggestion(self):
        plan = validate_resource_plan(
            "yolo",
            {"imgsz": 2048, "model_size": "n", "device": "0"},
            batch_size=16,
            profile=TEST_PROFILE,
        )
        self.assertFalse(plan["ok"])
        self.assertTrue(any("imgsz 2048" in item for item in plan["errors"]))
        self.assertTrue(plan["suggestions"])

    def test_workers_clamped_for_ram_profile(self):
        plan = validate_resource_plan(
            "resnet",
            {"image_size": 224, "workers": 12, "device": "0"},
            batch_size=16,
            profile=TEST_PROFILE,
        )
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["normalized_params"]["workers"], 4)
        self.assertTrue(plan["warnings"])

    def test_cpu_mode_uses_conservative_limits(self):
        plan = validate_resource_plan(
            "yolo",
            {"imgsz": 640, "model_size": "m", "workers": 8, "device": "cpu"},
            batch_size=8,
            profile=TEST_PROFILE,
        )
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["normalized_params"]["workers"], 2)

    @patch("resource_guard.subprocess.run", side_effect=FileNotFoundError())
    def test_resource_profile_schema_without_nvidia_smi(self, _mock_run):
        profile = get_resource_profile()
        self.assertIn("hardware", profile)
        self.assertIn("safe_limits", profile)
        self.assertEqual(profile["hardware"]["default_device"], "cpu")
        self.assertFalse(profile["hardware"]["gpu_available"])


if __name__ == "__main__":
    unittest.main()
