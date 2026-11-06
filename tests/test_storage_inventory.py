import os
from pathlib import Path
import tempfile
import unittest

from storage_inventory import category, ownership_roots, scan


class StorageInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('dataset', 'runs', 'storage', 'label-studio'):
            (self.root / name).mkdir()
        self.roots = {'dataset/alice': 'alice', 'runs/alice': 'alice', 'dataset/bob': 'bob'}
        for path in self.roots:
            (self.root / path).mkdir()

    def test_full_scan_hashes_files_counts_hardlinks_once_and_keeps_empty_legacy_dirs(self):
        path = self.root / 'dataset/alice/image'
        path.write_bytes(b'image')
        os.link(path, self.root / 'runs/alice/image')
        (self.root / 'dataset/unassigned-empty').mkdir()
        result = scan(self.root, self.roots)
        self.assertEqual(result['issues'], [])
        image = next(row for row in result['files'] if row['path'] == 'dataset/alice/image')
        self.assertEqual(len(image['sha256']), 64)
        self.assertEqual(result['breakdown']['alice']['datasets'], (path.stat().st_blocks + path.parent.stat().st_blocks) * 512)
        self.assertEqual(result['breakdown']['alice']['runs'], (self.root / 'runs/alice').stat().st_blocks * 512)

    def test_cross_owner_links_and_unassigned_files_fail_inventory(self):
        path = self.root / 'dataset/alice/image'
        path.write_bytes(b'image')
        os.link(path, self.root / 'dataset/bob/image')
        (self.root / 'storage/unknown').write_bytes(b'keep')
        reasons = {item['reason'] for item in scan(self.root, self.roots)['issues']}
        self.assertIn('Hard link crosses owners', reasons)
        self.assertIn('File has no verified owner', reasons)
        self.assertEqual(path.read_bytes(), b'image')

    def test_symlink_is_reported_without_following_it(self):
        (self.root / 'dataset/alice/link').symlink_to(self.root / 'dataset/bob', target_is_directory=True)
        issues = scan(self.root, self.roots)['issues']
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['path'], 'dataset/alice/link')

    def test_only_recognized_pid_lock_files_are_service_metadata(self):
        root = self.root / 'dataset/.locks'
        root.mkdir()
        lock = root / ('dataset-' + 'a' * 64 + '.lock')
        lock.write_text('1234')
        self.assertEqual(scan(self.root, self.roots)['issues'], [])
        lock.write_text('not a pid')
        self.assertEqual(scan(self.root, self.roots)['issues'][0]['reason'], 'File has no verified owner')

    def test_link_to_uninventoried_file_is_rejected(self):
        outside = self.root / 'outside'
        outside.write_bytes(b'x')
        os.link(outside, self.root / 'dataset/alice/image')
        self.assertTrue(any('escapes' in row['reason'] for row in scan(self.root, self.roots)['issues']))

    def test_mapping_rejects_wrong_owner_and_conflicting_paths(self):
        records = [{'storage_path': '/app/dataset/shared', 'owner_user_id': 'alice'},
                   {'storage_path': '/app/dataset/shared', 'owner_user_id': 'bob'}]
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            ownership_roots(['alice', 'bob'], records, [], [], [], [])
        with self.assertRaises(ValueError):
            ownership_roots(['alice'], [], [], [], [], [{'id': 1, 'owner': 'bob'}])

    def test_categories_distinguish_permanent_results_and_temporary_predictions(self):
        self.assertEqual(category('runs/run/evaluations/id/metrics.json'), 'evaluations')
        self.assertEqual(category('runs/.compute/id/input.png'), 'temporary')
        self.assertEqual(category('dataset/.owners/hash/.staged/token/dataset/image.png'), 'temporary')
        self.assertEqual(category('label-studio/media/upload/1/image.png'), 'annotation')
        self.assertEqual(category('dataset/.owners/hash/test/.ailab_index/gen/1.jpg'), 'cache')
