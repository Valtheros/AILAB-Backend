from __future__ import annotations

import asyncio
import io
import json
import stat
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import batch_prediction
from batch_inputs import extract_batch_zip
from batch_upload import stage_batch
from inference_service import InferenceError


class Upload:
    def __init__(self, filename, data):
        self.filename, self.file = filename, io.BytesIO(data)

    async def read(self, size):
        return self.file.read(size)


class BatchPredictionTests(unittest.TestCase):
    def test_zip_rejects_traversal_and_symbolic_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('../outside.jpg', '/absolute.jpg'):
                archive = root / 'input.zip'
                with zipfile.ZipFile(archive, 'w') as output:
                    output.writestr(name, b'image')
                with self.assertRaises(ValueError):
                    extract_batch_zip(archive, root)
            with zipfile.ZipFile(archive, 'w') as output:
                link = zipfile.ZipInfo('link.jpg')
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(link, 'target')
            with self.assertRaises(ValueError):
                extract_batch_zip(archive, root)

    def test_direct_uploads_are_bounded_and_do_not_share_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, digest, total = asyncio.run(stage_batch([
                Upload('same.jpg', b'a'), Upload('same.jpg', b'b'), Upload('bad.txt', b'c'),
            ], root))
            self.assertEqual(total, 3)
            self.assertEqual(len(digest), 64)
            self.assertEqual((root / 'inputs' / '0').read_bytes(), b'a')
            self.assertEqual((root / 'inputs' / '1').read_bytes(), b'b')
            self.assertIn('error', entries[2])
            with self.assertRaises(ValueError):
                asyncio.run(stage_batch([Upload(f'{i}.jpg', b'x') for i in range(101)], root))

    def test_batch_keeps_per_image_failure_and_escapes_csv_formulas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'job'
            asyncio.run(stage_batch([Upload('=name.jpg', b'valid'), Upload('bad.jpg', b'broken')], directory))
            row = {'id': uuid.uuid4(), 'dispatch_token': uuid.uuid4(), 'payload': {
                'project_name': 'run', 'task_type': 'image_classification', 'model_type': 'resnet', 'threshold': None,
            }}
            def predict(_run, payload, **_kwargs):
                return {'top': {'className': '@cat', 'confidence': .9}, 'predictions': []}
            def verify(payload):
                if payload == b'broken':
                    raise InferenceError('The uploaded file is not a readable image.')
                return MagicMock()
            context = MagicMock()
            with patch.object(batch_prediction, 'job_directory', return_value=directory), \
                 patch.object(batch_prediction, 'RUNS_DIR', root), \
                 patch.object(batch_prediction.registry, '_connect', return_value=context), \
                 patch('inference_service._open_verified_image', side_effect=verify), \
                 patch('inference_service.predict_image', side_effect=predict):
                batch_prediction.execute_batch(row, 'cpu')
            result = json.loads((directory / 'result.json').read_text())
            self.assertEqual((result['succeeded'], result['failed']), (1, 1))
            self.assertEqual(result['images'][1]['status'], 'error')
            self.assertIn("'=name.jpg", (directory / 'results.csv').read_text(encoding='utf-8-sig'))
            self.assertIn("'@cat", (directory / 'results.csv').read_text(encoding='utf-8-sig'))
            with zipfile.ZipFile(directory / 'results.zip') as archive:
                self.assertEqual(set(archive.namelist()), {'result.json', 'results.csv'})

    def test_model_failure_is_not_reported_as_an_image_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'job'
            asyncio.run(stage_batch([Upload('valid.jpg', b'valid')], directory))
            row = {'id': uuid.uuid4(), 'dispatch_token': uuid.uuid4(), 'payload': {
                'project_name': 'run', 'task_type': 'image_classification', 'model_type': 'resnet', 'threshold': None,
            }}
            with patch.object(batch_prediction, 'job_directory', return_value=directory), \
                 patch.object(batch_prediction, 'RUNS_DIR', root), \
                 patch.object(batch_prediction.registry, '_connect', return_value=MagicMock()), \
                 patch('inference_service._open_verified_image', return_value=MagicMock()), \
                 patch('inference_service.predict_image', side_effect=InferenceError('Model checkpoint is invalid.')):
                with self.assertRaisesRegex(InferenceError, 'Model checkpoint is invalid'):
                    batch_prediction.execute_batch(row, 'cpu')
            self.assertFalse((directory / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
