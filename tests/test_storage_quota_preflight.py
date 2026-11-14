import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.storage_quota_preflight import inspect


class StorageQuotaPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mount = {'target': '/', 'source': '/dev/test', 'fstype': 'ext4', 'options': 'rw,prjquota'}

    def inspect_mocked(self, *, mount=None, features=None, active=True):
        with patch('scripts.storage_quota_preflight._mount_info', return_value=mount or self.mount), \
             patch('scripts.storage_quota_preflight._filesystem_features', return_value=features), \
             patch('scripts.storage_quota_preflight.project_quota_active', return_value=active):
            return inspect([str(self.root)])

    def test_ready_filesystem_does_not_claim_application_enforcement(self):
        result = self.inspect_mocked(features={'project', 'quota'})
        self.assertTrue(result['filesystemsReady'])
        self.assertFalse(result['applicationEnforcementVerified'])

    def test_missing_features_or_mount_option_fail_closed(self):
        result = self.inspect_mocked(features={'quota'})
        self.assertFalse(result['filesystemsReady'])
        self.assertIn('features', result['errors'][0])
        result = self.inspect_mocked(mount={**self.mount, 'options': 'rw,relatime'}, features={'project', 'quota'})
        self.assertFalse(result['filesystemsReady'])
        self.assertIn('mount options', result['errors'][0])

    def test_missing_directory_and_unreadable_device_fail_closed(self):
        result = inspect([str(self.root / 'missing')])
        self.assertFalse(result['filesystemsReady'])
        with patch('scripts.storage_quota_preflight._mount_info', return_value=self.mount), \
             patch('scripts.storage_quota_preflight._filesystem_features', side_effect=PermissionError('denied')):
            result = inspect([str(self.root)])
        self.assertFalse(result['filesystemsReady'])
        self.assertIn('denied', result['errors'][0])

    def test_kernel_quota_probe_failure_fails_closed(self):
        with patch('scripts.storage_quota_preflight._mount_info', return_value=self.mount), \
             patch('scripts.storage_quota_preflight._filesystem_features', return_value={'project', 'quota'}), \
             patch('scripts.storage_quota_preflight.project_quota_active', side_effect=OSError('not active')):
            result = inspect([str(self.root)])
        self.assertFalse(result['filesystemsReady'])
        self.assertIn('not active', result['errors'][0])
        result = self.inspect_mocked(features={'project', 'quota'}, active=False)
        self.assertFalse(result['filesystemsReady'])
        self.assertIn('not active', result['errors'][0])

    def test_paths_on_different_filesystems_cannot_share_one_project_quota(self):
        other = self.root / 'other'
        other.mkdir()
        original_stat = Path.stat

        def stat_with_other_device(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if path == other:
                values = list(result)
                values[2] += 1
                return os.stat_result(values)
            return result

        with patch('scripts.storage_quota_preflight._mount_info', return_value=self.mount), \
             patch('scripts.storage_quota_preflight._filesystem_features', return_value={'project', 'quota'}), \
             patch('scripts.storage_quota_preflight.project_quota_active', return_value=True), \
             patch.object(Path, 'stat', stat_with_other_device):
            result = inspect([str(self.root), str(other)])
        self.assertFalse(result['filesystemsReady'])
        self.assertIn('different filesystems', result['errors'][0])


if __name__ == '__main__':
    unittest.main()
