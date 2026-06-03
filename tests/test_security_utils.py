from __future__ import annotations

import io
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {})

from dataset_utils import safe_dataset_name
from security_utils import contained_path, safe_extract_zip, validate_slug


class SecurityUtilsTests(unittest.TestCase):
    def test_slug_rejects_parent_directory(self):
        with self.assertRaises(ValueError):
            validate_slug("..", "dataset name")

    def test_slug_rejects_windows_reserved_name(self):
        with self.assertRaises(ValueError):
            validate_slug("CON", "dataset name")

    def test_dataset_name_falls_back_for_non_ascii_filename(self):
        self.assertEqual(safe_dataset_name("ข้อมูล.zip"), "dataset")

    def test_contained_path_rejects_sibling_prefix_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            with self.assertRaises(ValueError):
                contained_path(root, "../run-sibling/file.txt")

    def test_safe_extract_rejects_parent_directory(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../outside.txt", "unsafe")
        archive.seek(0)
        with tempfile.TemporaryDirectory() as temp:
            with zipfile.ZipFile(archive) as source, self.assertRaises(ValueError):
                safe_extract_zip(source, Path(temp) / "dataset")


if __name__ == "__main__":
    unittest.main()
