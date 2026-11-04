"""Check imports in the actual worker image from a job's working directory."""
import os
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.getenv('AILAB_WORKER_RUNTIME_TEST') == '1', 'Requires the built compute image')
class WorkerRuntimeTests(unittest.TestCase):
    def test_job_process_imports_application_modules(self):
        code = '''
import security_utils
assert security_utils.__file__ == "/app/security_utils.py", security_utils.__file__
import batch_prediction
import compute_executor
import dataset_utils
import evaluation_executor
import worker_app
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code], cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
