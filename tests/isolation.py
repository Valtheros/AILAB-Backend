"""Fail closed before integration tests touch a database, Redis, or media."""

import os
from pathlib import Path
from urllib.parse import urlsplit


def require_test_database(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {'postgres', 'postgresql'} or parsed.hostname not in {'127.0.0.1', 'localhost'}
            or parsed.username != 'ailab_test' or parsed.port not in {None, 5432}
            or parsed.path != '/ailab_compute_test_pg' or parsed.query or parsed.fragment):
        raise RuntimeError('Refusing to use a non-isolated PostgreSQL test database')


def require_test_redis(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'redis' or parsed.hostname not in {'127.0.0.1', 'localhost'}
            or parsed.port not in {None, 6379} or parsed.path != '/15' or parsed.query or parsed.fragment):
        raise RuntimeError('Refusing to use a non-isolated Redis test database')


def require_test_storage(*names):
    raw_root = os.environ.get('AILAB_TEST_ROOT', '')
    if not raw_root:
        raise RuntimeError('AILAB_TEST_ROOT is required for integration tests')
    root = Path(raw_root).resolve()
    marker = root / '.ailab-disposable-test'
    if not root.is_dir() or not marker.is_file() or marker.read_text().strip() != 'ailab-disposable-test-v1':
        raise RuntimeError('Refusing to use storage without the disposable test marker')
    for name in names:
        raw = os.environ.get(name, '')
        if not raw or not Path(raw).resolve().is_relative_to(root) or Path(raw).resolve() == root:
            raise RuntimeError(f'{name} must be inside AILAB_TEST_ROOT')
