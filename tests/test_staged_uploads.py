from __future__ import annotations

import tempfile
import json
import time
import unittest
from pathlib import Path

from dataset_storage import owner_dataset_root
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
            self.assertTrue(pending_stage.is_relative_to(owner_dataset_root(root / "datasets", "user-a")))
            loaded, dataset_dir, _ = store.consume(manifest["token"], "user-a")
            self.assertEqual(loaded["dataset_name"], "sample")
            self.assertTrue((dataset_dir / "data.txt").is_file())
            with self.assertRaises(FileNotFoundError):
                store.consume(manifest["token"], "user-a")

    def test_legacy_tokens_still_work_and_cleanup_preserves_consuming_data(self):
        with tempfile.TemporaryDirectory() as temp:
            store = StagedUploadStore(Path(temp) / "datasets")
            token = "legacy-token"
            stage = store.root / token
            data = stage / "dataset"
            data.mkdir(parents=True)
            (data / "data.txt").write_text("preserve", encoding="utf-8")
            manifest = {"token": token, "owner_id": "user-a", "dataset_name": "sample",
                        "expires_at": time.time() + 3600, "fingerprint": store._fingerprint(data)}
            (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(store.peek(token, "user-a")[2], stage)
            self.assertEqual(store.consuming_path(token, "user-a"), store.root / f".consuming-{token}")
            _, consumed, consuming = store.consume(token, "user-a")
            manifest["expires_at"] = 0
            (consuming / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            store.cleanup()
            self.assertEqual((consumed / "data.txt").read_text(encoding="utf-8"), "preserve")
            self.assertEqual(store.consuming_path(token, "user-a"), consuming)

    def test_cleanup_removes_only_expired_stages_with_valid_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            store = StagedUploadStore(Path(temp) / "datasets")
            root = store.owner_root("user-a")
            expired = root / "expired"
            malformed = root / "incomplete"
            expired.mkdir(parents=True)
            malformed.mkdir()
            (expired / "manifest.json").write_text(json.dumps({"owner_id": "user-a", "expires_at": 0}))
            (malformed / "data.txt").write_text("preserve")
            store.cleanup("user-a")
            self.assertFalse(expired.exists())
            self.assertEqual((malformed / "data.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
