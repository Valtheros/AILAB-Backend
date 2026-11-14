"""Integration checks require the dedicated disposable PostgreSQL database."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import evaluation_executor
from dataset_versioning import source_manifest, write_manifest
from resource_repository import resource_repository as registry
from tests.isolation import require_test_database, require_test_storage


@unittest.skipUnless(os.getenv('COMPUTE_TESTS') == '1', 'Requires isolated PostgreSQL test stack')
class EvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_test_database(registry.database_url)
        require_test_storage('RUNS_DIR', 'DATASET_DIR', 'STORAGE_DIR', 'TMPDIR')
        from scripts.apply_schema import apply
        with registry._connect() as connection:
            connection.execute('''create table if not exists "user"(id text primary key,email text unique,name text,
                "emailVerified" boolean default true,banned boolean default false,role text default 'user')''')
            apply(connection)

    def setUp(self):
        with registry._connect() as connection:
            connection.execute('truncate compute_jobs,compute_nodes,training_tasks,"user" cascade')
            connection.execute("insert into \"user\"(id,email) values('alice','alice@test.invalid')")

    def test_evaluation_persists_results_and_source_hash_in_postgres(self):
        import torch

        class Classifier(torch.nn.Module):
            def forward(self, tensor):
                return torch.tensor([[2., 0.]], device=tensor.device)

        class MutatingClassifier(Classifier):
            def __init__(self, path):
                super().__init__()
                self.path = path

            def forward(self, tensor):
                self.path.write_bytes(b'changed during evaluation')
                return super().forward(tensor)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets, runs = root / 'datasets', root / 'runs'
            source, run = datasets / 'sample', runs / 'run'
            (source / 'test' / 'cat').mkdir(parents=True)
            run.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(source / 'test' / 'cat' / 'image.png')
            metadata = {'classes': ['cat', 'dog'], 'formats': ['imagefolder'], 'tasks': ['image_classification']}
            digest = write_manifest(source, source_manifest(source, metadata))
            (run / 'best.pt').write_bytes(b'checkpoint')
            (run / 'job_config.json').write_text(json.dumps({'dataset_metadata': metadata}))
            dataset = registry.upsert_dataset('alice', 'alice@test.invalid', 'sample', str(source), metadata)
            evaluation_id, job_id, token = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            with registry._connect() as connection:
                connection.execute('update datasets set content_sha256=%s where id=%s', (digest, dataset['id']))
                run_id = connection.execute('''insert into training_tasks(owner_user_id,created_by,run_slug,task_type,model_type,status)
                    values('alice','alice','run','image_classification','resnet','completed') returning id''').fetchone()['id']
                payload = {'evaluation_id': str(evaluation_id)}
                row = connection.execute('''insert into compute_jobs(id,owner_user_id,run_id,kind,mode,status,payload,
                    dataset_version_id,dispatch_token) values(%s,'alice',%s,'evaluate','cpu','running',%s::jsonb,%s,%s) returning *''',
                    (job_id, run_id, json.dumps(payload), dataset['id'], token)).fetchone()
                connection.execute('''insert into evaluations(id,owner_user_id,run_id,dataset_version_id,compute_job_id,
                    origin,status,dataset_content_sha256) values(%s,'alice',%s,%s,%s,'manual','running',%s)''',
                    (evaluation_id, run_id, dataset['id'], job_id, digest))
            entry = {'model': Classifier(), 'classes': ['cat', 'dog'], 'image_size': 8, 'device': 'cpu'}
            with patch.object(evaluation_executor, 'DATASET_DIR', datasets), patch.object(evaluation_executor, 'RUNS_DIR', runs), \
                 patch('inference_service.load_classifier', return_value=entry):
                checkpoint = run / 'best.pt'
                image_path = source / 'test' / 'cat' / 'image.png'
                original_image = image_path.read_bytes()
                for path, message in ((checkpoint, 'Checkpoint changed'), (image_path, 'Dataset changed')):
                    entry['model'] = MutatingClassifier(path)
                    with self.assertRaisesRegex(ValueError, message):
                        evaluation_executor.evaluate_job(row, 'cpu')
                    self.assertFalse((run / 'evaluations' / f'.{evaluation_id}-{token}').exists())
                    checkpoint.write_bytes(b'checkpoint')
                    image_path.write_bytes(original_image)
                entry['model'] = Classifier()
                result, status = evaluation_executor.evaluate_job(row, 'cpu')
            report = json.loads(Path(result).read_text())
            self.assertEqual(status, 'completed')
            self.assertTrue(report['sourceVerified'])
            self.assertEqual(report['datasetContentSha256'], digest)
            self.assertEqual(report['metrics']['accuracy'], 1.0)
            with registry._connect() as connection:
                image = connection.execute('select * from evaluation_images where evaluation_id=%s', (evaluation_id,)).fetchone()
                self.assertEqual(image['outcome'], 'correct')
                self.assertEqual(connection.execute('select progress_current from compute_jobs where id=%s', (job_id,)).fetchone()['progress_current'], 1)


if __name__ == '__main__':
    unittest.main()
