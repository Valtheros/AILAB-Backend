import unittest
from unittest.mock import patch

from storage_project_quota import ProjectQuotaMeasurement, assign_empty_directory, set_project_limit


class StorageProjectQuotaValidationTests(unittest.TestCase):
    def test_invalid_project_ids_do_not_open_any_directory(self):
        with patch('storage_project_quota._open_directory') as opened:
            for project_id in (0, -1, True, 4294967295, 1.0, '1'):
                with self.subTest(project_id=project_id), self.assertRaises(ValueError):
                    assign_empty_directory('/not-used', project_id)
        opened.assert_not_called()

    def test_invalid_hard_limits_do_not_access_the_device(self):
        with patch('storage_project_quota._validate_device') as validated:
            for limit in (0, -1024, True, 1, 1025, 2**64):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    set_project_limit('/not-used', '/not-used', 1, limit)
        validated.assert_not_called()

    def test_failed_anchor_open_closes_filesystem_root(self):
        measure = ProjectQuotaMeasurement('/device', '/anchors', '/root')
        with patch('storage_project_quota._open_directory', side_effect=[41, OSError('unavailable')]), \
                patch('storage_project_quota.os.close') as close:
            with self.assertRaises(OSError):
                measure(1)
        close.assert_called_once_with(41)


if __name__ == '__main__':
    unittest.main()
