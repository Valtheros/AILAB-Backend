from __future__ import annotations

import unittest

from model_catalog import validate_model_params


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


if __name__ == "__main__":
    unittest.main()
