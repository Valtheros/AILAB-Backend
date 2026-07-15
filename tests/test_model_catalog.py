from __future__ import annotations

import unittest

from model_catalog import get_catalog, validate_model_params


class ModelCatalogValidationTests(unittest.TestCase):
    def test_rejects_unknown_model_param(self):
        with self.assertRaises(ValueError):
            validate_model_params("yolo", {"unexpected": True})

    def test_rejects_param_above_catalog_maximum(self):
        with self.assertRaises(ValueError):
            validate_model_params("yolo", {"workers": 100})

    def test_accepts_cataloged_params(self):
        self.assertEqual(validate_model_params("yolo", {"workers": 4, "cache": False}), {"workers": 4, "cache": False})

    def test_rejects_non_positive_batch_size(self):
        with self.assertRaises(ValueError):
            validate_model_params("yolo", {"batch_size": -1})


    def test_catalog_models_expose_dataset_interface_metadata(self):
        catalog = get_catalog()
        for task in catalog["tasks"]:
            for model in task["models"]:
                with self.subTest(model=model["id"]):
                    self.assertTrue(model.get("dataset_task"))
                    self.assertTrue(model.get("required_annotations"))
                    self.assertTrue(model.get("accepted_canonical_formats"))
                    self.assertTrue(model.get("train_export_format"))


    def test_catalog_models_expose_resource_guardrail_metadata(self):
        catalog = get_catalog()
        for task in catalog["tasks"]:
            for model in task["models"]:
                with self.subTest(model=model["id"]):
                    self.assertTrue(model.get("safe_defaults"))
                    self.assertTrue(model.get("hard_limits"))
                    self.assertTrue(model.get("memory_notes"))
                    self.assertEqual(model.get("resource_profile", {}).get("policy"), "auto_safe")

    def test_catalog_exposes_only_supported_vision_tasks(self):
        catalog = get_catalog()
        self.assertEqual(
            {task["id"] for task in catalog["tasks"]},
            {"image_classification", "segmentation", "object_detection"},
        )


if __name__ == "__main__":
    unittest.main()
