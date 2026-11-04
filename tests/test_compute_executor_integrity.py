from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from api_errors import PublicError
import compute_executor


class ComputeExecutorIntegrityTests(unittest.TestCase):
    def test_changed_checkpoint_fails_job_after_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'job'
            directory.mkdir()
            (directory / 'input').write_bytes(b'image')
            (root / 'run').mkdir()
            job_id, token = uuid.uuid4(), uuid.uuid4()
            row = {'id': job_id, 'kind': 'predict', 'payload': {'project_name': 'run', 'task_type': 'image_classification',
                'model_type': 'resnet', 'checkpoint_path': 'best.pt', 'checkpoint_sha256': 'hash'}}
            connection = MagicMock()
            connection.__enter__.return_value.execute.return_value.fetchone.return_value = row
            changed = PublicError('CHECKPOINT_CHANGED', 'The saved model changed.', status=409)
            with patch.object(compute_executor, 'RUNS_DIR', root), patch.object(compute_executor, 'job_directory', return_value=directory), \
                 patch.object(compute_executor.registry, '_connect', return_value=connection), \
                 patch.dict('os.environ', {'COMPUTE_EXEC_DEVICE': 'cpu'}), \
                 patch('model_api_service.verify_checkpoint', side_effect=[None, changed]), \
                 patch('inference_service.predict_image', return_value={'predictions': []}):
                compute_executor.run(str(job_id), str(token))
            outcome = json.loads((directory / f'{token}.json').read_text())
            self.assertEqual(outcome['status'], 'failed')
            self.assertEqual(outcome['error_code'], 'CHECKPOINT_CHANGED')

    def test_training_verifies_dataset_before_and_after_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'job'
            directory.mkdir()
            job_id, token = uuid.uuid4(), uuid.uuid4()
            row = {'id': job_id, 'kind': 'train', 'payload': {'source_dataset_path': str(root),
                   'dataset_content_sha256': 'digest', 'extra_args': {}}}
            connection = MagicMock()
            connection.__enter__.return_value.execute.return_value.fetchone.return_value = row
            trainer = MagicMock()
            with patch.object(compute_executor, 'job_directory', return_value=directory), \
                 patch.object(compute_executor.registry, '_connect', return_value=connection), \
                 patch.object(compute_executor, 'registered_storage_path', return_value=root), \
                 patch.object(compute_executor, 'verify_source', side_effect=[True, True]) as verify, \
                 patch.dict('sys.modules', {'worker_app': SimpleNamespace(run_training=trainer)}), \
                 patch.dict('os.environ', {'COMPUTE_EXEC_DEVICE': 'cpu'}):
                compute_executor.run(str(job_id), str(token))
            outcome = json.loads((directory / f'{token}.json').read_text())
            self.assertEqual(outcome['status'], 'completed')
            self.assertTrue(outcome['dataset_verified'])
            self.assertEqual(verify.call_count, 2)
            trainer.assert_called_once()

    def test_changed_training_dataset_fails_without_claiming_verified_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'job'
            directory.mkdir()
            job_id, token = uuid.uuid4(), uuid.uuid4()
            row = {'id': job_id, 'kind': 'train', 'payload': {'source_dataset_path': str(root),
                   'dataset_content_sha256': 'digest', 'extra_args': {}}}
            connection = MagicMock()
            connection.__enter__.return_value.execute.return_value.fetchone.return_value = row
            with patch.object(compute_executor, 'job_directory', return_value=directory), \
                 patch.object(compute_executor.registry, '_connect', return_value=connection), \
                 patch.object(compute_executor, 'registered_storage_path', return_value=root), \
                 patch.object(compute_executor, 'verify_source', side_effect=[True, ValueError('Dataset changed')]), \
                 patch.dict('sys.modules', {'worker_app': SimpleNamespace(run_training=MagicMock())}), \
                 patch.dict('os.environ', {'COMPUTE_EXEC_DEVICE': 'cpu'}):
                compute_executor.run(str(job_id), str(token))
            outcome = json.loads((directory / f'{token}.json').read_text())
            self.assertEqual(outcome['status'], 'failed')
            self.assertNotIn('dataset_verified', outcome)


if __name__ == '__main__':
    unittest.main()
