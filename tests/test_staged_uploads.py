from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from staged_uploads import StagedUploadStore


class StagedUploadTests(unittest.TestCase):
    def test_token_is_owner_bound_and_single_stage_can_be_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "data.txt").write_text("data", encoding="utf-8")
            store = StagedUploadStore(root / "datasets")
            manifest = store.create("user-a", "sample", source, {"formats": ["imagefolder"]})
            with self.assertRaises(FileNotFoundError):
                store.consume(manifest["token"], "user-b")
            peeked, pending_dataset, pending_stage = store.peek(manifest["token"], "user-a")
            self.assertEqual(peeked["dataset_name"], "sample")
            self.assertTrue(pending_dataset.is_dir())
            self.assertTrue(pending_stage.is_dir())
            loaded, dataset_dir, _ = store.consume(manifest["token"], "user-a")
            self.assertEqual(loaded["dataset_name"], "sample")
            self.assertTrue((dataset_dir / "data.txt").is_file())
            with self.assertRaises(FileNotFoundError):
                store.consume(manifest["token"], "user-a")


if __name__ == "__main__":
    unittest.main()
