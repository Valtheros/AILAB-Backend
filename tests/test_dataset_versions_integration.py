"""Dataset version tests use only the dedicated disposable PostgreSQL database."""
from __future__ import annotations

import os
import asyncio
import io
import json
import tempfile
import unittest
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from starlette.datastructures import UploadFile
from starlette.requests import Request

from dataset_storage import owner_dataset_root
from dataset_utils import inspect_dataset_for_upload
from resource_repository import resource_repository as registry
from staged_uploads import StagedUploadStore
from tests.isolation import require_test_database, require_test_storage


@unittest.skipUnless(os.getenv('COMPUTE_TESTS') == '1', 'Requires isolated PostgreSQL test stack')
class DatasetVersionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_test_database(registry.database_url)
        require_test_storage('RUNS_DIR', 'DATASET_DIR', 'STORAGE_DIR', 'APP_BASE_DIR', 'TMPDIR')
        global main
        import main
        from scripts.apply_schema import apply
        with registry._connect() as connection:
            connection.execute('''create table if not exists "user"(id text primary key,email text unique,name text,
                                  "emailVerified" boolean default true,banned boolean default false,role text default 'user')''')
            apply(connection)
            apply(connection)

    def setUp(self):
        with registry._connect() as connection:
            connection.execute('truncate dataset_import_intents,dataset_collections,datasets,training_tasks,"user" cascade')
            connection.execute("insert into \"user\"(id,email) values('alice','alice@example.invalid'),('bob','bob@example.invalid')")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = StagedUploadStore(self.root)
        for name, value in [('DATASET_DIR', self.root), ('staged_uploads', self.store)]:
            current = patch.object(main, name, value)
            current.start()
            self.addCleanup(current.stop)

    def stage(self, owner='alice', colour=1):
        root = self.root / f'extracted-{os.urandom(8).hex()}'
        for index, category in enumerate(('cat', 'dog')):
            path = root / 'train' / category
            path.mkdir(parents=True)
            Image.new('RGB', (8, 8), (colour, index, 0)).save(path / 'image.png')
        metadata = inspect_dataset_for_upload(root)
        return self.store.create(owner, 'pictures', root, metadata)['token']

    @staticmethod
    def request(owner='alice'):
        return Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': [
            (b'x-user-id', owner.encode()), (b'x-user-email', f'{owner}@example.invalid'.encode()),
        ]})

    def test_imports_create_distinct_versions_without_overwriting(self):
        first = main.import_dataset(main.StagedImportRequest(uploadToken=self.stage()), self.request())
        original = registry.get_dataset('alice', first['dataset_name'])
        source = Path(original['storage_path']) / 'train' / 'cat' / 'image.png'
        content = source.read_bytes()
        second = main.import_dataset(main.StagedImportRequest(uploadToken=self.stage(colour=2),
                                      collectionId=first['collectionId']), self.request())
        self.assertEqual((first['version'], second['version']), (1, 2))
        self.assertNotEqual(first['id'], second['id'])
        self.assertNotEqual(first['contentSha256'], second['contentSha256'])
        self.assertEqual(source.read_bytes(), content)
        with self.assertRaises(main.HTTPException) as error:
            main.import_dataset(main.StagedImportRequest(uploadToken=self.stage('bob'),
                                collectionId=first['collectionId']), self.request('bob'))
        self.assertEqual(error.exception.status_code, 404)

    def test_dataset_collection_requires_a_real_owner(self):
        from psycopg.errors import ForeignKeyViolation

        with registry._connect() as connection:
            with self.assertRaises(ForeignKeyViolation):
                with connection.transaction():
                    connection.execute("insert into dataset_collections(owner_user_id,name) values('missing-user','test')")

    def test_retry_after_database_failure_recovers_moved_files_once(self):
        payload = main.StagedImportRequest(uploadToken=self.stage())
        with patch.object(registry, 'insert_dataset_version', side_effect=RuntimeError('simulated disconnect')):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                main.import_dataset(payload, self.request())
        recovered = main.import_dataset(payload, self.request())
        repeated = main.import_dataset(payload, self.request())
        self.assertEqual(recovered['id'], repeated['id'])
        self.assertEqual(len(registry.list_datasets('alice')), 1)

    def test_real_zip_inspection_stages_under_the_owner_and_imports(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w') as archive:
            for index, name in enumerate(('cat', 'dog')):
                image = io.BytesIO()
                Image.new('RGB', (8, 8), (index, 0, 0)).save(image, format='PNG')
                archive.writestr(f'train/{name}/image.png', image.getvalue())
        content.seek(0)
        upload = UploadFile(filename='pictures.zip', file=content)
        inspected = asyncio.run(main._inspect_uploaded_zip(self.request(), upload))
        token = inspected['uploadToken']
        stage = self.store.peek(token, 'alice')[2]
        self.assertTrue(stage.is_relative_to(owner_dataset_root(self.root, 'alice')))
        imported = main.import_dataset(main.StagedImportRequest(uploadToken=token), self.request())
        record = registry.get_dataset('alice', imported['dataset_name'])
        self.assertTrue((Path(record['storage_path']) / 'train' / 'cat' / 'image.png').is_file())

    def test_legacy_interrupted_import_can_recover_after_stage_expiry(self):
        token = self.stage()
        stage = self.store.peek(token, 'alice')[2]
        stage.replace(self.store.root / token)
        payload = main.StagedImportRequest(uploadToken=token)
        with patch.object(registry, 'insert_dataset_version', side_effect=RuntimeError('simulated disconnect')):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                main.import_dataset(payload, self.request())
        consuming = self.store.root / f'.consuming-{token}'
        manifest_path = consuming / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['expires_at'] = 0
        manifest_path.write_text(json.dumps(manifest))
        self.store.cleanup()
        self.assertTrue(consuming.exists())
        imported = main.import_dataset(payload, self.request())
        self.assertIsNotNone(registry.get_dataset('alice', imported['dataset_name']))

    def test_pending_import_blocks_account_operations_before_label_studio_cleanup(self):
        payload = main.StagedImportRequest(uploadToken=self.stage())
        with patch.object(registry, 'insert_dataset_version', side_effect=RuntimeError('simulated disconnect')):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                main.import_dataset(payload, self.request())
        with self.assertRaisesRegex(ValueError, 'dataset import is unfinished'):
            registry.transfer_user_resources('alice', 'bob')
        with self.assertRaisesRegex(ValueError, 'dataset import is unfinished'):
            registry.delete_user_resource_records('alice')
        with patch.object(main, '_require_admin'), patch('label_studio_jobs.delete_account_data') as cleanup:
            with self.assertRaises(main.HTTPException) as error:
                main.admin_delete_user_resources('alice', self.request('bob'))
            self.assertEqual(error.exception.status_code, 409)
            cleanup.assert_not_called()
        recovered = main.import_dataset(payload, self.request())
        self.assertIsNotNone(registry.get_dataset('alice', recovered['dataset_name']))

    def test_disabled_account_cannot_finish_a_pending_import(self):
        payload = main.StagedImportRequest(uploadToken=self.stage())
        with patch.object(registry, 'insert_dataset_version', side_effect=RuntimeError('simulated disconnect')):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                main.import_dataset(payload, self.request())
        with registry._connect() as connection:
            connection.execute('update "user" set banned=true where id=%s', ('alice',))
        with self.assertRaises(main.HTTPException) as error:
            main.import_dataset(payload, self.request())
        self.assertEqual(error.exception.status_code, 403)

    def test_concurrent_uploads_reserve_different_versions(self):
        first = main.import_dataset(main.StagedImportRequest(uploadToken=self.stage()), self.request())
        payloads = [main.StagedImportRequest(uploadToken=self.stage(colour=colour), collectionId=first['collectionId'])
                    for colour in (2, 3)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            versions = list(pool.map(lambda payload: main.import_dataset(payload, self.request())['version'], payloads))
        self.assertEqual(sorted(versions), [2, 3])
        self.assertEqual(len(registry.list_datasets('alice')), 3)

    def test_draft_pins_dataset_id_and_run_records_manifest(self):
        imported = main.import_dataset(main.StagedImportRequest(uploadToken=self.stage()), self.request())
        draft = registry.create_task('alice', 'alice@example.invalid')
        saved = registry.update_task_draft('alice', draft['id'], {
            'display_name': 'test', 'task_type': 'image_classification', 'model_type': 'resnet',
            'model_name': 'resnet18', 'dataset_slug': imported['dataset_name'], 'params': {},
        })
        self.assertEqual(str(saved['dataset_id']), imported['id'])
        with registry.dataset_guard('alice', imported['dataset_name']) as connection:
            dataset = registry.get_dataset('alice', imported['dataset_name'], connection)
            activated = registry.activate_task(owner_id='alice', task_id=draft['id'], dataset=dataset,
                run_slug='test-run', task_type='image_classification', model_type='resnet', model_name='resnet18',
                params={}, storage_path=self.root / 'run', job_id='test-job', connection=connection)
        self.assertEqual(activated['dataset_content_sha256'], imported['contentSha256'])
        self.assertEqual(activated['dataset_provenance'], 'unverified')

    def test_backfill_preserves_legacy_identity_and_existing_run_provenance(self):
        from scripts import backfill_dataset_versions as backfill
        source = self.root / 'legacy'
        source.mkdir()
        (source / 'image.png').write_bytes(b'legacy')
        record = registry.upsert_dataset('alice', None, 'legacy', source, {'classes': ['x']})
        with registry._connect() as connection:
            run = connection.execute('''insert into training_tasks(owner_user_id,created_by,dataset_id,dataset_slug,
                task_type,model_type,status) values('alice','alice',%s,'legacy','image_classification','resnet','completed')
                returning id''', (record['id'],)).fetchone()
        with patch.object(backfill, 'DATASET_DIR', self.root):
            self.assertEqual(backfill.backfill()['updated'], 1)
            self.assertFalse((source / '.ailab_source_manifest.json').exists())
            self.assertEqual(backfill.backfill(True)['updated'], 1)
            self.assertEqual(backfill.backfill(True)['updated'], 0)
        updated = registry.get_dataset('alice', 'legacy')
        self.assertEqual(updated['id'], record['id'])
        self.assertEqual(updated['storage_path'], str(source))
        self.assertEqual(updated['version'], 1)
        self.assertTrue(updated['content_sha256'])
        self.assertEqual(registry.get_task('alice', run['id'])['dataset_provenance'], 'unverified')

    def test_label_studio_versions_share_counter_with_zip_imports(self):
        from scripts import backfill_dataset_versions as backfill
        project_id, operation_id = uuid.uuid4(), uuid.uuid4()
        with registry._connect() as connection:
            connection.execute("insert into label_studio_accounts(owner_user_id,ls_user_id) values('alice',12)")
            connection.execute('''insert into label_studio_projects(id,owner_user_id,ls_project_id,name,kind,reserved_version)
                values(%s,'alice',12,'Label project','object_detection',3)''', (project_id,))
            connection.execute('''insert into label_studio_operations(id,owner_user_id,project_id,kind,status,version)
                values(%s,'alice',%s,'publish','completed',3)''', (operation_id, project_id))
        source = self.root / 'legacy-ls'
        source.mkdir()
        (source / 'image.png').write_bytes(b'legacy')
        registry.upsert_dataset('alice', None, 'legacy-ls', source, {'label_studio_operation': str(operation_id)})
        with patch.object(backfill, 'DATASET_DIR', self.root):
            backfill.backfill(True)
        dataset = registry.get_dataset('alice', 'legacy-ls')
        self.assertEqual(dataset['version'], 3)
        with registry.dataset_guard('alice', 'other') as connection:
            collection = registry.reserve_dataset_version('alice', 'other', dataset['collection_id'], connection)
        self.assertEqual(collection['reserved_version'], 4)
        with registry._connect() as connection:
            project = connection.execute('select reserved_version from label_studio_projects where id=%s', (project_id,)).fetchone()
        self.assertEqual(project['reserved_version'], 4)


if __name__ == '__main__':
    unittest.main()
