from __future__ import annotations

import ast
import io
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path


def _safe_load_stub(text: str):
    data = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("["):
            try:
                data[key.strip()] = ast.literal_eval(value)
            except Exception:
                data[key.strip()] = []
        else:
            data[key.strip()] = value
    return data


def _dump_stub(data, *_args, **_kwargs):
    return "\n".join(f"{key}: {value}" for key, value in data.items()) + "\n"


if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=_safe_load_stub, dump=_dump_stub)

from dataset_utils import safe_dataset_name
from security_utils import contained_path, named_file_lock, safe_extract_zip, validate_slug


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


    def test_named_file_lock_rejects_concurrent_same_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with named_file_lock(root, "shared_dataset", "dataset name"):
                with self.assertRaises(FileExistsError):
                    with named_file_lock(root, "shared_dataset", "dataset name"):
                        pass
            self.assertFalse((root / ".locks" / "shared_dataset.lock").exists())

    def test_safe_extract_rejects_parent_directory(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../outside.txt", "unsafe")
        archive.seek(0)
        with tempfile.TemporaryDirectory() as temp:
            with zipfile.ZipFile(archive) as source, self.assertRaises(ValueError):
                safe_extract_zip(source, Path(temp) / "dataset")

    def test_safe_extract_treats_zero_byte_implied_directory_as_directory(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("smoke_yolo", "")
            output.writestr("smoke_yolo/images/train/image.jpg", b"image")
        archive.seek(0)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "dataset"
            with zipfile.ZipFile(archive) as source:
                safe_extract_zip(source, target)
            self.assertTrue((target / "smoke_yolo" / "images" / "train" / "image.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
