import os
from pathlib import Path
import tempfile
import unittest

from storage_reconciliation import breakdown


class StorageReconciliationTests(unittest.TestCase):
    def test_counts_allocated_blocks_and_deduplicates_hardlinks_by_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('dataset', 'runs', 'storage', 'label-studio'):
                (root / name).mkdir()
            path = root / 'dataset/source'
            path.write_bytes(b'image')
            os.link(path, root / 'runs/link')
            owned = path.stat().st_ino
            result = breakdown(root, {1: 'alice', 2: 'bob'},
                               lambda fd: [0, 0, 0, 1 if os.fstat(fd).st_ino == owned else 0])
            self.assertEqual(result, {'alice': {'datasets': path.stat().st_blocks * 512}, 'bob': {}})
            (root / 'dataset/link').symlink_to(root / 'runs', target_is_directory=True)
            with self.assertRaises(ValueError):
                breakdown(root, {}, lambda _fd: [0, 0, 0, 0])
