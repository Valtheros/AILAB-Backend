import tempfile
import unittest
from pathlib import Path

from dataset_storage import owner_storage_key
from scripts.storage_ownership_audit import audit


class StorageOwnershipAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets, self.runs, self.uploads = (self.root / name for name in ('datasets', 'runs', 'uploads'))
        for path in (self.datasets, self.runs, self.uploads):
            path.mkdir()
        self.owner_key = owner_storage_key('alice')
        (self.datasets / '.owners' / self.owner_key / 'dataset').mkdir(parents=True)
        (self.datasets / '.annotations' / self.owner_key / 'legacy').mkdir(parents=True)
        (self.runs / 'run').mkdir()
        (self.runs / '.compute' / 'job-1').mkdir(parents=True)
        (self.uploads / '7').mkdir()
        self.records = {'owners': ['alice', 'bob'],
                        'datasets': [{'owner_user_id': 'alice', 'storage_path': str(self.datasets / '.owners' / self.owner_key / 'dataset')}],
                        'runs': [{'owner_user_id': 'alice', 'storage_path': str(self.runs / 'run')}],
                        'projects': [{'owner_user_id': 'alice', 'ls_project_id': 7}],
                        'jobs': [{'owner_user_id': 'alice', 'id': 'job-1'}]}

    def inspect(self):
        return audit(self.datasets, self.runs, self.uploads, **self.records)

    def test_maps_legacy_and_label_studio_without_claiming_full_verification(self):
        result = self.inspect()
        self.assertFalse(result['ownershipVerified'])
        self.assertEqual(result['issues'], [])
        self.assertEqual((result['counts']['.annotations'], result['counts']['label-studio'], result['counts']['compute']), (1, 1, 1))
        self.assertTrue((self.datasets / '.annotations' / self.owner_key / 'legacy').is_dir())

    def test_reports_unknown_entries_conflicting_owners_and_symlinks(self):
        (self.uploads / 'unknown').mkdir()
        (self.runs / 'link').symlink_to(self.runs / 'run', target_is_directory=True)
        self.records['runs'].append({'owner_user_id': 'bob', 'storage_path': str(self.runs / 'run')})
        issues = self.inspect()['issues']
        reasons = {issue['reason'] for issue in issues}
        self.assertIn('conflicting owners', reasons)
        self.assertIn('no unambiguous owner', reasons)
        self.assertIn('symbolic link needs review', reasons)

    def test_dataset_path_for_another_owner_is_not_trusted(self):
        self.records['datasets'][0]['owner_user_id'] = 'bob'
        issues = self.inspect()['issues']
        self.assertTrue(any(issue['reason'] == 'dataset path and owner do not match' for issue in issues))

    def test_records_pointing_to_missing_users_are_not_assigned(self):
        self.records['projects'][0]['owner_user_id'] = 'removed-user'
        self.records['runs'][0]['owner_user_id'] = 'removed-user'
        issues = self.inspect()['issues']
        self.assertTrue(any(issue['reason'] == 'missing owner' for issue in issues))
        self.assertTrue(any(issue['reason'] == 'run owner is not in the user table' for issue in issues))

    def test_legacy_root_symlink_is_not_followed(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.datasets / '.annotation-uploads').symlink_to(outside, target_is_directory=True)
        issues = self.inspect()['issues']
        self.assertTrue(any(issue['category'] == '.annotation-uploads' and issue['reason'] == 'symbolic link needs review'
                            for issue in issues))
