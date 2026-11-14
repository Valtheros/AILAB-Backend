import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from fastapi import HTTPException
from starlette.requests import Request

import main
from api_errors import PublicError


class QaRegressionTests(unittest.TestCase):
    def test_query_artifact_checks_owner_and_path(self):
        request = Request({'type': 'http', 'headers': [(b'x-user-id', b'alice')]})
        other = Request({'type': 'http', 'headers': [(b'x-user-id', b'bob')]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run').mkdir()
            (root / 'run' / 'train.log').write_text('private log')
            with patch.object(main, 'RUNS_DIR', root), patch.object(main, '_run_owner', return_value='alice'):
                self.assertEqual(main.download_run_file_query('run', request, 'train.log').path, root / 'run' / 'train.log')
                with self.assertRaises(HTTPException) as denied:
                    main.download_run_file_query('run', other, 'train.log')
                self.assertEqual(denied.exception.status_code, 404)
                with self.assertRaises(HTTPException) as escaped:
                    main.download_run_file_query('run', request, '../outside')
                self.assertEqual(escaped.exception.status_code, 404)

    def test_compute_required_rejects_before_legacy_queue(self):
        with patch.dict(os.environ, {'COMPUTE_REQUIRED': 'true', 'COMPUTE_ENABLED': 'false'}), patch.object(main.training_service, 'start_training_container') as start:
            with self.assertRaises(PublicError) as error:
                main._start_train_request(None, None)
            self.assertEqual(error.exception.code, 'COMPUTE_DISABLED')
            start.assert_not_called()

    def test_label_studio_classification_display_uses_declared_labels(self):
        metadata = {'formats': ['imagefolder'], 'tasks': ['image_classification'], 'classes': ['000_curve', '001_junction'],
                    'image_count': 2, 'size_bytes': 2, 'warnings': [], 'errors': [],
                    'label_schema': {'origin': 'label_studio', 'task': 'image_classification', 'classes': ['curve', 'junction']}}
        profile = main._dataset_profile(metadata)
        self.assertEqual(profile['classes'], ['curve', 'junction'])
        self.assertEqual(metadata['classes'], ['000_curve', '001_junction'])

    def test_train_again_names_the_copy_without_changing_original(self):
        request = Request({'type': 'http', 'headers': [(b'x-user-id', b'alice')]})
        original = {'id': 'old', 'display_name': 'run', 'status': 'completed', 'params': {}}
        draft = {'id': '12345678-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}
        with patch.object(main.resource_repository, 'get_task', return_value=original), \
             patch.object(main.resource_repository, 'create_task', return_value=draft), \
             patch.object(main.resource_repository, 'update_task_draft', side_effect=lambda _owner, _id, values: values), \
             patch.object(main, '_task_response', side_effect=lambda row: row):
            result = main.clone_task('old', request)
        self.assertRegex(result['display_name'], r'^run \(copy \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC - 12345678\)$')
        self.assertEqual(original['display_name'], 'run')

    def test_metrics_keep_data_without_hardcoded_interpretation(self):
        request = Request({'type': 'http', 'headers': [(b'x-user-id', b'alice')]})
        rows = [{'epoch': '1', 'train/accuracy': '0.5', 'val/accuracy': '1.0'}]
        with patch.object(main, '_run_owner', return_value='alice'), \
             patch.object(main.training_service, 'get_training_metrics', return_value=rows):
            result = main.get_metrics('run', request)
        self.assertEqual(result['metrics'], rows)
        self.assertEqual(result['insights'], [])

    def test_compare_returns_metrics_without_hardcoded_interpretation(self):
        request = Request({'type': 'http', 'headers': [(b'x-user-id', b'alice')]})
        ids = ['01234567-0000-0000-0000-000000000001', '01234567-0000-0000-0000-000000000002']
        rows = [{'id': run_id, 'run_slug': f'run-{index}', 'dataset_slug': 'data'} for index, run_id in enumerate(ids)]
        with patch.object(type(main.resource_repository), 'enabled', new_callable=PropertyMock, return_value=True), \
             patch.object(main.resource_repository, 'get_task', side_effect=rows), \
             patch.object(main.training_service, 'get_training_metrics', return_value=[]), \
             patch.object(main, '_read_json_file', return_value={}), \
             patch.object(main, 'build_run_comparison', side_effect=lambda task_row, **_kwargs: task_row), \
             patch.object(main, 'summarise_comparison', return_value={'warnings': []}):
            result = main.compare_dataset_runs('data', ','.join(ids), request)
        self.assertEqual(len(result['runs']), 2)
        self.assertEqual(result['insights'], [])
