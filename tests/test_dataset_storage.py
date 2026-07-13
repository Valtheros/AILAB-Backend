from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_storage import dataset_lock_name, owner_dataset_path, owner_dataset_root, registered_storage_path


class DatasetStorageTests(unittest.TestCase):
    def test_same_dataset_slug_is_isolated_per_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "datasets"
            owner_a = owner_dataset_path(root, "user-a", "car-park.coco")
            owner_b = owner_dataset_path(root, "user-b", "car-park.coco")

            self.assertNotEqual(owner_a, owner_b)
            self.assertEqual(owner_a.name, owner_b.name)
            self.assertTrue(owner_a.is_relative_to(root.resolve()))
            self.assertTrue(owner_b.is_relative_to(root.resolve()))

            owner_a.mkdir(parents=True)
            owner_b.mkdir(parents=True)
            (owner_a / "owner.txt").write_text("user-a", encoding="utf-8")
            (owner_b / "owner.txt").write_text("user-b", encoding="utf-8")

            self.assertEqual((owner_a / "owner.txt").read_text(encoding="utf-8"), "user-a")
            self.assertEqual((owner_b / "owner.txt").read_text(encoding="utf-8"), "user-b")

    def test_owner_roots_and_locks_are_stable_and_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "datasets"
            self.assertEqual(owner_dataset_root(root, "user-a"), owner_dataset_root(root, "user-a"))
            self.assertNotEqual(owner_dataset_root(root, "user-a"), owner_dataset_root(root, "user-b"))
            self.assertNotEqual(
                dataset_lock_name("user-a", "shared"),
                dataset_lock_name("user-b", "shared"),
            )

    def test_registered_storage_must_remain_under_dataset_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "datasets"
            expected = owner_dataset_path(root, "user-a", "sample")
            self.assertEqual(registered_storage_path(root, expected), expected)
            with self.assertRaises(ValueError):
                registered_storage_path(root, root.parent / "outside")


if __name__ == "__main__":
    unittest.main()
