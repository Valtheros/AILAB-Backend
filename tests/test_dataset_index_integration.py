"""Dataset gallery integration requires isolated PostgreSQL and Redis."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from tests.isolation import require_test_database, require_test_storage


@unittest.skipUnless(os.getenv('COMPUTE_TESTS') == '1', 'Requires isolated PostgreSQL and Redis')
class DatasetIndexIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from resource_repository import resource_repository
        cls.registry = resource_repository
        require_test_database(cls.registry.database_url)
        require_test_storage('RUNS_DIR', 'DATASET_DIR', 'STORAGE_DIR', 'APP_BASE_DIR', 'TMPDIR')
        from scripts.apply_schema import apply
        with cls.registry._connect() as connection:
            connection.execute('''create table if not exists "user"(id text primary key,email text unique,name text,
                "emailVerified" boolean default true,banned boolean default false,role text default 'user')''')
            apply(connection)
        from main import app
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    def setUp(self):
        from settings import DATASET_DIR
        with self.registry._connect() as connection:
            connection.execute('truncate compute_jobs,compute_nodes,training_tasks,"user" cascade')
            connection.execute("insert into \"user\"(id,email) values('alice','alice@test.invalid'),('bob','bob@test.invalid')")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=DATASET_DIR)
        self.root = Path(self.temp.name)
        for split in ('train', 'test'):
            folder = self.root / split / 'cat'
            folder.mkdir(parents=True)
            Image.new('RGB', (600, 300), 'red').save(folder / 'image_1.png')
        self.dataset = self.registry.upsert_dataset('alice', 'alice@test.invalid', 'index-test', self.root,
            {'formats': ['imagefolder'], 'tasks': ['image_classification'], 'classes': ['cat']})

    def tearDown(self):
        self.temp.cleanup()

    def headers(self, user):
        return {'x-internal-token': 'integration-token', 'x-user-id': user}

    def test_gallery_uses_thumbnails_summary_and_owner_scoping(self):
        from dataset_index_jobs import run
        dataset_id = self.dataset['id']
        base = f'/api/dataset-versions/{dataset_id}'
        self.assertEqual(self.client.get(base, headers=self.headers('bob')).status_code, 404)
        self.assertEqual(self.client.post(base + '/index', headers=self.headers('bob')).status_code, 404)
        response = self.client.post(base + '/index', headers=self.headers('alice'))
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(len(self.registry.active_runs_for_dataset(dataset_id)), 1)
        with self.registry._connect() as connection:
            generation = connection.execute('select working_generation from dataset_indexes where dataset_version_id=%s',
                                            (dataset_id,)).fetchone()['working_generation']
        run(str(dataset_id), str(generation))
        run(str(dataset_id), str(generation))
        details = self.client.get(base, headers=self.headers('alice')).json()
        self.assertEqual(details['index']['status'], 'completed')
        self.assertEqual(details['index']['stats']['train']['classes']['cat']['images'], 1)
        response = self.client.get(base + '/images', params={'split': 'train', 'search': 'image_1', 'className': 'cat'}, headers=self.headers('alice'))
        self.assertEqual(response.status_code, 200, response.text)
        listing = response.json()
        self.assertEqual(listing['total'], 1)
        self.assertNotIn('annotations', listing['images'][0])
        self.assertNotIn('source_path', listing['images'][0])
        ordinal = listing['images'][0]['ordinal']
        self.assertEqual(self.client.get(f'{base}/images/{ordinal}', headers=self.headers('alice')).json()['annotations']['className'], 'cat')
        thumbnail = self.client.get(f'{base}/images/{ordinal}/thumbnail', headers=self.headers('alice'))
        self.assertEqual(thumbnail.status_code, 200)
        self.assertEqual(thumbnail.headers['cache-control'], 'private, no-store')
        self.assertEqual(self.client.get(f'{base}/images/{ordinal}/source', headers=self.headers('bob')).status_code, 404)
        duplicates = self.client.get(base + '/duplicates', headers=self.headers('alice')).json()['groups']
        self.assertEqual(len(duplicates), 1)
        self.assertEqual({item['split'] for item in duplicates[0]['images']}, {'train', 'test'})
        duplicate_path = f"{base}/duplicates/{duplicates[0]['hash']}"
        self.assertEqual(self.client.get(duplicate_path, headers=self.headers('alice')).json()['total'], 2)
        self.assertEqual(self.client.get(duplicate_path, headers=self.headers('bob')).status_code, 404)

    def test_worker_restart_requeues_incomplete_generation_without_touching_source(self):
        from dataset_index_jobs import recover, run
        dataset_id = self.dataset['id']
        base = f'/api/dataset-versions/{dataset_id}'
        self.client.post(base + '/index', headers=self.headers('alice'))
        with self.registry._connect() as connection:
            old = connection.execute("update dataset_indexes set status='running' where dataset_version_id=%s returning working_generation",
                                     (dataset_id,)).fetchone()['working_generation']
            connection.execute('update dataset_index_generations set cache_created=true where dataset_version_id=%s and generation=%s',
                               (dataset_id, old))
        old_cache = self.root / '.ailab_index' / str(old)
        old_cache.mkdir(parents=True)
        (old_cache / 'partial.jpg').write_bytes(b'generated')
        recover()
        with self.registry._connect() as connection:
            row = connection.execute('select * from dataset_indexes where dataset_version_id=%s', (dataset_id,)).fetchone()
        self.assertEqual(row['status'], 'queued')
        self.assertNotEqual(row['working_generation'], old)
        run(str(dataset_id), str(row['working_generation']))
        self.assertEqual(self.client.get(base, headers=self.headers('alice')).json()['index']['status'], 'completed')
        self.assertFalse(old_cache.exists())
        self.assertTrue((self.root / 'train' / 'cat' / 'image_1.png').is_file())

    def test_index_runs_through_the_cpu_rq_queue(self):
        from dataset_index_jobs import QUEUE
        from redis import Redis
        from rq import Queue, SimpleWorker
        from settings import REDIS_URL

        redis = Redis.from_url(REDIS_URL)
        queue = Queue(QUEUE, connection=redis)
        queue.empty()
        base = f"/api/dataset-versions/{self.dataset['id']}"
        self.assertEqual(self.client.post(base + '/index', headers=self.headers('alice')).status_code, 202)
        SimpleWorker([queue], connection=redis).work(burst=True, logging_level='WARNING')
        details = self.client.get(base, headers=self.headers('alice')).json()
        self.assertEqual(details['index']['status'], 'completed')
        self.assertEqual(self.client.get(base + '/images', headers=self.headers('alice')).json()['total'], 2)


if __name__ == '__main__':
    unittest.main()
