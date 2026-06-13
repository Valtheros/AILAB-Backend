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

    def test_paddleocr_catalog_exposes_base_model_presets(self):
        catalog = get_catalog()
        ocr_task = next(task for task in catalog["tasks"] if task["id"] == "ocr")
        paddleocr = next(model for model in ocr_task["models"] if model["id"] == "paddleocr")
        presets = {preset["id"]: preset for preset in paddleocr["base_model_presets"]}

        self.assertEqual(presets["ppocrv4-rec"]["task"], "rec")
        self.assertIn("rec_gt_train.txt", presets["ppocrv4-rec"]["labels"])
        self.assertEqual(presets["ppocrv4-det"]["task"], "det")
        self.assertIn("det_gt_train.txt", presets["ppocrv4-det"]["labels"])

    def test_tesseract_catalog_exposes_start_model_presets(self):
        catalog = get_catalog()
        ocr_task = next(task for task in catalog["tasks"] if task["id"] == "ocr")
        tesseract = next(model for model in ocr_task["models"] if model["id"] == "tesseract")
        presets = {preset["value"]: preset for preset in tesseract["base_model_presets"]}

        self.assertTrue(presets["eng"]["available"])
        self.assertTrue(presets["tha"]["available"])


if __name__ == "__main__":
    unittest.main()
