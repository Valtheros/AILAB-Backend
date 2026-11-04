"""Optional CPU or CUDA smoke test with disposable data and real trainers."""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

DEVICE = os.getenv('AILAB_REAL_MODEL_DEVICE', 'cpu')


@unittest.skipUnless(os.getenv('AILAB_REAL_MODEL_TEST') == '1', 'Requires compute image and real model dependencies')
class RealClassificationSmokeTests(unittest.TestCase):
    def test_train_evaluate_and_batch_resnet_and_efficientnet(self):
        import torch
        import batch_prediction
        from batch_upload import stage_batch
        from evaluation_executor import _evaluate_classification
        from tests.test_batch_prediction import Upload
        from worker.trainers.classification_common import train_classifier

        self.assertIn(DEVICE, ('cpu', 'cuda:0'))
        if DEVICE == 'cuda:0':
            self.assertTrue(torch.cuda.is_available(), 'CUDA is not available for the requested smoke test')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, runs = root / 'dataset', root / 'runs'
            for split in ('train', 'val', 'test'):
                for index, name in enumerate(('cat', 'dog')):
                    folder = source / split / name
                    folder.mkdir(parents=True)
                    Image.new('RGB', (32, 32), (index * 120, 32, 96)).save(folder / 'one.png')
            metadata = {'classes': ['cat', 'dog'], 'formats': ['imagefolder'], 'tasks': ['image_classification']}
            for family, architecture in (('resnet', 'resnet18'), ('efficientnet', 'efficientnet_b0')):
                with self.subTest(model=family), patch.dict(os.environ, {'RUNS_DIR': str(runs)}):
                    config = {'dataset_path': str(source), 'dataset_metadata': metadata, 'task_type': 'image_classification',
                              'model_type': family, 'model_name': architecture, 'project_name': family,
                              'epochs': 1, 'batch_size': 2, 'compute_job_id': str(uuid.uuid4()),
                              'extra_args': {'architecture': architecture, 'pretrained': False, 'image_size': 32,
                                             'freeze_backbone': True, 'workers': 0, 'device': '0' if DEVICE == 'cuda:0' else 'cpu', 'amp': False}}
                    train_classifier(config, family, None, lambda *_: None)
                    self.assertTrue((runs / family / 'best.pt').is_file())
                    writer = MagicMock()
                    with patch('evaluation_executor._record_progress'):
                        report = _evaluate_classification({}, source, metadata, runs / family, config, writer, DEVICE)
                    self.assertEqual((report['status'], report['evaluated'], report['skipped']), ('completed', 2, 0))
                    self.assertEqual({item['name'] for item in report['metrics']['classes']}, {'cat', 'dog'})
                    image = io.BytesIO()
                    Image.new('RGB', (32, 32)).save(image, format='PNG')
                    directory = root / f'{family}-batch'
                    asyncio.run(stage_batch([Upload('image.png', image.getvalue()), Upload('bad.jpg', b'bad')], directory))
                    row = {'id': uuid.uuid4(), 'dispatch_token': uuid.uuid4(), 'payload': {
                        'project_name': family, 'task_type': 'image_classification', 'model_type': family, 'threshold': None}}
                    with patch.object(batch_prediction, 'job_directory', return_value=directory), \
                         patch.object(batch_prediction, 'RUNS_DIR', runs), patch.object(batch_prediction, '_write_progress'):
                        batch_prediction.execute_batch(row, DEVICE)
                    result = json.loads((directory / 'result.json').read_text())
                    self.assertEqual((result['succeeded'], result['failed']), (1, 1))
                    self.assertEqual(result['images'][0]['result']['model']['device'], DEVICE)


if __name__ == '__main__':
    unittest.main()
