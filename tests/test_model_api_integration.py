"""Public model API checks against isolated disposable PostgreSQL and Redis."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from tests.isolation import require_test_database, require_test_redis, require_test_storage


@unittest.skipUnless(os.getenv('COMPUTE_TESTS') == '1', 'Requires isolated PostgreSQL and Redis')
class ModelAPIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import model_api_service
        cls.quota_gate = patch.object(model_api_service, 'STORAGE_ENFORCEMENT_IMPLEMENTED', True)
        cls.quota_gate.start()
        cls.addClassCleanup(cls.quota_gate.stop)
        from resource_repository import resource_repository
        cls.registry = resource_repository
        from settings import REDIS_URL
        require_test_database(cls.registry.database_url)
        require_test_redis(REDIS_URL)
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
        from redis import Redis
        from settings import REDIS_URL, RUNS_DIR
        Redis.from_url(REDIS_URL).flushdb()
        with self.registry._connect() as connection:
            connection.execute('truncate compute_jobs,compute_nodes,training_tasks,"user" cascade')
            connection.execute("insert into \"user\"(id,email) values('alice','alice@test.invalid'),('bob','bob@test.invalid')")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='model-api-test-', dir=RUNS_DIR))
        self.runs = {}
        for subject in ('alice', 'bob'):
            run_dir = self.directory / subject
            run_dir.mkdir()
            (run_dir / 'best.pt').write_bytes(b'test checkpoint')
            (run_dir / 'job_config.json').write_text(json.dumps({'extra_args': {}}))
            # Run slugs are single path components in the application.
            target = RUNS_DIR / f'{self.directory.name}-{subject}'
            run_dir.replace(target)
            with self.registry._connect() as connection:
                run_id = connection.execute('''insert into training_tasks(owner_user_id,created_by,run_slug,task_type,model_type,status)
                    values(%s,%s,%s,'image_classification','resnet','completed') returning id''',
                    (subject, subject, target.name)).fetchone()['id']
            self.runs[subject] = {'id': run_id, 'slug': target.name, 'path': target}

    def tearDown(self):
        for run in self.runs.values():
            shutil.rmtree(run['path'], ignore_errors=True)
        shutil.rmtree(self.directory, ignore_errors=True)

    def session(self, subject):
        return {'x-internal-token': 'integration-token', 'x-user-id': subject}

    def endpoint(self, subject):
        response = self.client.post('/api/model-endpoints', headers=self.session(subject), json={'runSlug': self.runs[subject]['slug']})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['id']

    def key(self, subject, endpoint):
        response = self.client.post(f'/api/model-endpoints/{endpoint}/keys', headers=self.session(subject), json={'label': 'test'})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['key']['id'], response.json()['secret']

    def submit(self, endpoint, secret, request_key='first', headers=None):
        image = io.BytesIO()
        Image.new('RGB', (8, 8)).save(image, format='PNG')
        return self.client.post(f'/api/v1/models/{endpoint}/predict', headers={
            'authorization': f'Bearer {secret}', 'idempotency-key': request_key, **(headers or {})},
            data={'execution': '{"mode":"cpu"}'}, files={'file': ('image.png', image.getvalue(), 'image/png')})

    def test_key_scope_idempotency_disable_and_revocation(self):
        import compute_repository
        alice = self.endpoint('alice')
        bob = self.endpoint('bob')
        key_id, secret = self.key('alice', alice)
        _, second = self.key('alice', alice)
        with self.registry._connect() as connection:
            stored = connection.execute('select key_sha256 from model_api_keys where id=%s', (key_id,)).fetchone()
        self.assertEqual(stored['key_sha256'], hashlib.sha256(secret.encode()).hexdigest())
        self.assertNotEqual(stored['key_sha256'], secret)
        self.assertEqual(self.submit(bob, secret, headers={'x-user-id': 'bob'}).status_code, 404)
        created = self.submit(alice, secret, headers={'x-user-id': 'bob'})
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()['id']
        self.assertEqual(self.submit(alice, secret).json()['id'], job_id)
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}', headers={'authorization': f'Bearer {second}'}).status_code, 404)
        directory = compute_repository.job_directory(job_id)
        (directory / 'result.json').write_text(json.dumps({'predictions': [{'className': 'cat'}]}))
        with self.registry._connect() as connection:
            connection.execute("update compute_jobs set status='completed',finished_at=now() where id=%s", (job_id,))
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/result', headers={'authorization': f'Bearer {secret}'}).status_code, 200)
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/result', headers={'authorization': f'Bearer {second}'}).status_code, 404)
        with self.registry._connect() as connection:
            with self.assertRaises(Exception):
                connection.execute('delete from training_tasks where id=%s', (self.runs['alice']['id'],))
        self.assertEqual(self.client.patch(f'/api/model-endpoints/{alice}', headers=self.session('alice'), json={'enabled': False}).status_code, 200)
        self.assertEqual(self.submit(alice, secret, 'another').status_code, 409)
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}', headers={'authorization': f'Bearer {secret}'}).status_code, 200)
        with self.registry._connect() as connection:
            connection.execute("update compute_jobs set finished_at=now()-interval '25 hours' where id=%s", (job_id,))
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/result', headers={'authorization': f'Bearer {secret}'}).status_code, 410)
        self.assertEqual(self.client.post(f'/api/model-endpoints/{alice}/keys/{key_id}/revoke', headers=self.session('alice')).status_code, 200)
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}', headers={'authorization': f'Bearer {secret}'}).status_code, 401)

    def test_key_expiry_ban_and_checkpoint_change_are_enforced(self):
        endpoint = self.endpoint('alice')
        key_id, secret = self.key('alice', endpoint)
        with self.registry._connect() as connection:
            connection.execute("update model_api_keys set expires_at=now()-interval '1 second' where id=%s", (key_id,))
        self.assertEqual(self.submit(endpoint, secret).status_code, 401)
        with self.registry._connect() as connection:
            connection.execute("update model_api_keys set expires_at=now()+interval '1 day' where id=%s", (key_id,))
            connection.execute('update "user" set banned=true where id=%s', ('alice',))
        self.assertEqual(self.submit(endpoint, secret).status_code, 401)
        with self.registry._connect() as connection:
            connection.execute('update "user" set banned=false where id=%s', ('alice',))
        (self.runs['alice']['path'] / 'best.pt').write_bytes(b'changed checkpoint')
        self.client.patch(f'/api/model-endpoints/{endpoint}', headers=self.session('alice'), json={'enabled': False})
        response = self.client.patch(f'/api/model-endpoints/{endpoint}', headers=self.session('alice'), json={'enabled': True})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['code'], 'CHECKPOINT_CHANGED')

    def test_rate_limit_returns_retry_after_and_blocks_spoofed_identity(self):
        endpoint = self.endpoint('alice')
        _, secret = self.key('alice', endpoint)
        for _ in range(20):
            self.assertEqual(self.submit(endpoint, secret).status_code, 202)
        response = self.submit(endpoint, secret, headers={'x-user-id': 'bob'})
        self.assertEqual(response.status_code, 429)
        self.assertGreater(int(response.headers['retry-after']), 0)
        self.assertEqual(self.client.get('/api/v1/jobs/' + str(uuid.uuid4()), headers={'x-user-id': 'alice', 'x-internal-token': 'integration-token'}).status_code, 401)

    def test_invalid_batch_returns_client_error_and_does_not_create_a_job(self):
        from settings import RUNS_DIR
        endpoint = self.endpoint('alice')
        _, secret = self.key('alice', endpoint)
        headers = {'authorization': f'Bearer {secret}', 'idempotency-key': 'invalid-batch'}
        compute_root = RUNS_DIR / '.compute'
        before = set(compute_root.iterdir()) if compute_root.exists() else set()
        response = self.client.post(f'/api/v1/models/{endpoint}/predict-batch', headers=headers,
                                    data={'execution': '{"mode":"cpu"}'},
                                    files={'files': ('broken.zip', b'not a zip', 'application/zip')})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()['code'], 'INVALID_BATCH')
        self.assertIn('ZIP', response.json()['detail'])
        with self.registry._connect() as connection:
            count = connection.execute('select count(*) as n from compute_jobs').fetchone()['n']
        self.assertEqual(count, 0)
        self.assertEqual(set(compute_root.iterdir()) if compute_root.exists() else set(), before)

    def test_batch_submission_is_idempotent_and_results_are_key_scoped(self):
        import compute_repository

        endpoint = self.endpoint('alice')
        _, secret = self.key('alice', endpoint)
        _, other_secret = self.key('alice', endpoint)
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(image, format='PNG')
        headers = {'authorization': f'Bearer {secret}', 'idempotency-key': 'batch-once'}
        def submit(content):
            return self.client.post(f'/api/v1/models/{endpoint}/predict-batch', headers=headers,
                                    data={'execution': '{"mode":"cpu"}'},
                                    files={'files': ('image.png', content, 'image/png')})

        created = submit(image.getvalue())
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()['id']
        self.assertEqual(submit(image.getvalue()).json()['id'], job_id)
        self.assertEqual(submit(image.getvalue() + b'different').status_code, 409)
        result_url = f'/api/v1/jobs/{job_id}/result'
        self.assertEqual(self.client.get(result_url, headers=headers).status_code, 409)
        self.assertEqual(self.client.get(result_url, headers={'authorization': f'Bearer {other_secret}'}).status_code, 404)

        directory = compute_repository.job_directory(job_id)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        (directory / 'result.json').write_text(json.dumps({'succeeded': 1, 'failed': 0, 'images': []}))
        (directory / 'results.csv').write_text('filename,status\nimage.png,completed\n')
        with self.registry._connect() as connection:
            connection.execute("update compute_jobs set status='completed',finished_at=now() where id=%s", (job_id,))
        response = self.client.get(result_url, headers=headers)
        self.assertEqual((response.status_code, response.json()['succeeded']), (200, 1))
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/files/results.csv', headers=headers).status_code, 200)
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/files/results.csv',
                                         headers={'authorization': f'Bearer {other_secret}'}).status_code, 404)
        with self.registry._connect() as connection:
            connection.execute("update compute_jobs set finished_at=now()-interval '25 hours' where id=%s", (job_id,))
        self.assertEqual(self.client.get(result_url, headers=headers).status_code, 410)
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/files/results.csv', headers=headers).status_code, 410)
