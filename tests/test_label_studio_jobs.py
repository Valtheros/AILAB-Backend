import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import label_studio_jobs as jobs


class QueueDurabilityTests(unittest.TestCase):
    def test_validation_error_is_visible_but_server_details_are_private(self):
        message = 'Image 403 is unfinished. Submit its annotation or exclude it before publishing.'
        self.assertEqual(jobs.operation_error(jobs.BridgeError(message, 400)), message)
        for error in (jobs.BridgeError('secret database credentials', 503), RuntimeError('secret database credentials')):
            self.assertNotIn('secret', jobs.operation_error(error))
        self.assertLessEqual(len(jobs.operation_error(jobs.BridgeError('x' * 1000, 400))), 500)

    def test_preview_requires_owner_and_completed_state(self):
        context = MagicMock()
        connection = context.__enter__.return_value
        with patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs.registry, 'upsert_dataset') as register:
            connection.execute.return_value.fetchone.return_value = None
            with self.assertRaises(FileNotFoundError):
                jobs.accept_preview('other-user', 'preview')
            connection.execute.return_value.fetchone.return_value = {'preview_only': True, 'result': {}, 'status': 'running'}
            with self.assertRaises(ValueError):
                jobs.accept_preview('owner', 'preview')
            register.assert_not_called()

    def test_discard_removes_only_preview_without_registering_dataset(self):
        context = MagicMock()
        connection = context.__enter__.return_value
        connection.execute.return_value.fetchone.return_value = {'id': 'preview', 'preview_only': True, 'result': {}, 'status': 'completed'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            preview = root / '.label-studio-previews' / 'preview'
            preview.mkdir(parents=True)
            original = root / 'original.png'
            original.touch()
            with patch.object(jobs, 'DATASET_DIR', root), patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs.registry, 'upsert_dataset') as register:
                self.assertEqual(jobs.accept_preview('owner', 'preview', discard=True), {'discarded': True})
                register.assert_not_called()
            self.assertFalse(preview.exists())
            self.assertTrue(original.exists())

    def test_accept_is_idempotent(self):
        context = MagicMock()
        result = {'accepted': True, 'datasetId': 'dataset'}
        context.__enter__.return_value.execute.return_value.fetchone.return_value = {'preview_only': True, 'result': result}
        with patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs.registry, 'upsert_dataset') as register:
            self.assertEqual(jobs.accept_preview('owner', 'preview'), result)
            register.assert_not_called()

    def test_job_identity_is_committed_before_redis_enqueue(self):
        events = []
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = {'id': 'operation'}
        context = MagicMock()
        context.__enter__.return_value = connection
        context.__exit__.side_effect = lambda *args: events.append('commit')
        queue = MagicMock()
        queue.enqueue.side_effect = lambda *args, **kwargs: events.append('enqueue')
        with patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs, 'Queue', return_value=queue):
            jobs.enqueue('operation')
        self.assertEqual(events, ['commit', 'enqueue'])

    def test_cancelled_operation_is_not_enqueued(self):
        context = MagicMock()
        context.__enter__.return_value.execute.return_value.fetchone.return_value = None
        queue = MagicMock()
        with patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs, 'Queue', return_value=queue):
            jobs.enqueue('operation')
        queue.enqueue.assert_not_called()

    def test_operation_is_not_run_twice_when_database_lock_is_held(self):
        context = MagicMock()
        context.__enter__.return_value.execute.return_value.fetchone.return_value = {'acquired': False}
        with patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs, '_run') as run:
            jobs.run('operation')
        run.assert_not_called()

    def test_repeated_worker_crashes_do_not_retry_forever(self):
        context = MagicMock()
        connection = context.__enter__.return_value
        connection.execute.return_value.fetchall.return_value = [{
            'id': 'operation', 'status': 'running', 'rq_job_id': None, 'attempt': 3, 'cancel_requested': False,
        }]
        connection.execute.return_value.fetchone.return_value = {'acquired': True}
        with patch.object(jobs.registry, '_connect', return_value=context), patch.object(jobs, 'Redis'), patch.object(jobs, 'enqueue') as enqueue:
            jobs.recover()
        enqueue.assert_not_called()
        self.assertTrue(any('Worker stopped repeatedly' in call.args[0] for call in connection.execute.call_args_list))
