import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.isolation import require_test_database, require_test_redis, require_test_storage


class IsolationTests(unittest.TestCase):
    def test_database_requires_exact_loopback_test_target(self):
        require_test_database('postgresql://ailab_test:password@127.0.0.1:5432/ailab_compute_test_pg')
        for url in ('postgresql://ailab_test@db:5432/ailab_compute_test_pg',
                    'postgresql://ailab_test@127.0.0.1/ailab_compute_test_pg_production',
                    'postgresql://admin@localhost/ailab_compute_test_pg',
                    'postgresql://ailab_test@localhost/ailab_compute_test_pg?host=production'):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                require_test_database(url)

    def test_redis_requires_dedicated_loopback_database(self):
        require_test_redis('redis://127.0.0.1:6379/15')
        for url in ('redis://redis:6379/15', 'redis://localhost:6379/0',
                    'redis://localhost:6379/15?db=0', 'rediss://localhost:6379/15'):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                require_test_redis(url)

    def test_storage_requires_marker_and_contained_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {'AILAB_TEST_ROOT': str(root), 'RUNS_DIR': str(root / 'runs')}):
                with self.assertRaises(RuntimeError):
                    require_test_storage('RUNS_DIR')
                (root / '.ailab-disposable-test').write_text('ailab-disposable-test-v1')
                require_test_storage('RUNS_DIR')
                with patch.dict(os.environ, {'RUNS_DIR': str(root.parent)}), self.assertRaises(RuntimeError):
                    require_test_storage('RUNS_DIR')
                with patch.dict(os.environ, {'RUNS_DIR': str(root)}), self.assertRaises(RuntimeError):
                    require_test_storage('RUNS_DIR')
