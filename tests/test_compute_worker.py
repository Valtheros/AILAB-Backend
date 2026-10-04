import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import compute_executor
import compute_worker


class WorkerFailureTests(unittest.TestCase):
    def supervisor(self, *, code=None, cancel=False, timeout=False, kill_code=-15, child_result=None):
        process = Mock(returncode=code)
        process.poll.side_effect = lambda: process.returncode
        row = {'kind': 'predict', 'worker_cpus': 4}
        connection = Mock()
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)

        def stop():
            if process.returncode is None:
                process.returncode = kill_code

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(compute_worker, 'LEASE', Mock()), \
             patch.object(compute_worker, 'GPU', None), \
             patch.object(compute_worker.registry, '_connect', return_value=connection), \
             patch.object(compute_worker, 'claim_job', return_value=row), \
             patch.object(compute_worker, 'job_directory', return_value=Path(temp)), \
             patch.object(compute_worker.subprocess, 'Popen', return_value=process), \
             patch.object(compute_worker, 'heartbeat', return_value={'cancel_requested': cancel}), \
             patch.object(compute_worker, 'terminate_child', side_effect=stop), \
             patch.object(compute_worker, 'finish_job') as finish, \
             patch.object(compute_worker.threading, 'Thread'), \
             patch.object(compute_worker.time, 'monotonic', side_effect=[0, 0, 1, 1000, 1000] if timeout else None, return_value=0):
            if child_result:
                (Path(temp) / 'token.json').write_text(json.dumps(child_result))
            outcome = compute_worker.execute('job', 'token')
            self.assertTrue((Path(temp) / 'released-token.json').is_file())
            finish.assert_called_once_with('job', 'token', **outcome)
            return outcome

    def test_killed_process_releases_only_after_exit(self):
        self.assertEqual(self.supervisor(code=-9)['error_code'], 'WORKER_KILLED')

    def test_cancel_finishes_after_child_stops(self):
        self.assertEqual(self.supervisor(cancel=True)['status'], 'cancelled')

    def test_timeout_is_reported(self):
        self.assertEqual(self.supervisor(timeout=True)['error_code'], 'COMPUTE_TIMEOUT')

    def test_forced_stop_reason_is_preserved_after_sigkill(self):
        self.assertEqual(self.supervisor(timeout=True, kill_code=-9)['error_code'], 'COMPUTE_TIMEOUT')
        self.assertEqual(self.supervisor(cancel=True, kill_code=-9)['status'], 'cancelled')

    def test_late_child_result_does_not_override_timeout(self):
        self.assertEqual(self.supervisor(timeout=True, child_result={'status': 'completed'})['error_code'], 'COMPUTE_TIMEOUT')

    def test_directory_failure_finishes_without_release_file(self):
        for error in (OSError(28, 'No space left on device'), PermissionError('Read-only directory')):
            with self.subTest(error=error):
                directory = MagicMock()
                directory.mkdir.side_effect = error
                directory.__truediv__.return_value.write_text.side_effect = error
                self.assertEqual(self.setup_failure(directory=directory)['error_code'], 'WORKER_STORAGE_ERROR')

    def setup_failure(self, *, directory=None, start_error=None, finish_error=None, clear=True, duplicate=False, claim_error=None, commit_error=None):
        connection = MagicMock()
        connection.__exit__.side_effect = commit_error
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(compute_worker, 'LEASE', Mock()), \
             patch.object(compute_worker, 'CHILD', None), \
             patch.object(compute_worker, 'GPU', None), \
             patch.object(compute_worker.registry, '_connect', return_value=connection), \
             patch.object(compute_worker, 'claim_job', return_value=None if duplicate else {'kind': 'predict', 'worker_cpus': 4}, side_effect=claim_error), \
             patch.object(compute_worker, 'job_directory', return_value=directory or Path(temp)), \
             patch.object(compute_worker.subprocess, 'Popen', side_effect=start_error or OSError('Cannot spawn process')) as start, \
             patch.object(compute_worker, 'gpu_clear', return_value=clear), \
             patch.object(compute_worker, 'finish_job', side_effect=finish_error) as finish, \
             patch.object(compute_worker.threading, 'Thread'), \
             patch.object(compute_worker, 'log'), \
             patch.object(compute_worker.time, 'sleep'), \
             patch.object(compute_worker.os, '_exit', side_effect=SystemExit) as exit_worker:
            if finish_error or not clear or claim_error or commit_error:
                with self.assertRaises(SystemExit):
                    compute_worker.execute('job', 'token')
                exit_worker.assert_called_once_with(1)
                if not clear or claim_error or commit_error:
                    finish.assert_not_called()
                return
            result = compute_worker.execute('job', 'token')
            if duplicate:
                finish.assert_not_called()
                start.assert_not_called()
            else:
                finish.assert_called_once_with('job', 'token', **result)
            exit_worker.assert_not_called()
            return result

    def test_process_start_failure_finishes(self):
        self.assertEqual(self.setup_failure(start_error=OSError('Cannot fork'))['status'], 'failed')

    def test_release_proof_failure_does_not_block_db_update(self):
        with patch.object(Path, 'write_text', side_effect=OSError(28, 'Disk full')):
            self.assertEqual(self.setup_failure()['status'], 'failed')

    def test_finish_db_failure_exits_for_replacement_recovery(self):
        self.setup_failure(finish_error=ConnectionError('DB unavailable'))

    def test_claim_failure_exits_instead_of_leaving_an_unwatched_job(self):
        self.setup_failure(claim_error=ConnectionError('Commit acknowledgement lost'))

    def test_claim_commit_failure_exits_without_guessing_its_state(self):
        self.setup_failure(commit_error=ConnectionError('Commit acknowledgement lost'))

    def test_gpu_not_clear_keeps_reservation(self):
        self.setup_failure(clear=False)

    def test_duplicate_delivery_never_finalizes_other_job(self):
        self.assertEqual(self.setup_failure(duplicate=True), {'duplicate': True})

    def test_executor_reports_oom_without_public_traceback(self):
        connection = Mock()
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)
        connection.execute.return_value.fetchone.return_value = {'kind': 'train', 'payload': {}}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(compute_executor, 'job_directory', return_value=Path(temp)), \
             patch.object(compute_executor.registry, '_connect', return_value=connection), \
             patch.dict('os.environ', {'COMPUTE_EXEC_DEVICE': 'cpu'}), \
             patch.dict('sys.modules', {'worker_app': SimpleNamespace(
                 run_training=Mock(side_effect=RuntimeError('CUDA out of memory')))}):
            compute_executor.run('job', 'token')
            result = json.loads((Path(temp) / 'token.json').read_text())
            self.assertEqual(result['error_code'], 'GPU_OOM')
            self.assertNotIn('Traceback', result['error_message'])
            self.assertTrue((Path(temp) / 'error.log').is_file())
