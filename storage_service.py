"""Application entry points for quota-bound files and short-lived request leases."""
from __future__ import annotations

import fcntl
import errno
from contextlib import contextmanager
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from resource_repository import resource_repository as registry
from settings import STORAGE_DIR
from storage_reservations import StorageReservationError, StorageReservations


log = logging.getLogger(__name__)
ledger = StorageReservations(registry._connect)
_COMPUTE_OUTPUT_BYTES = {'train': 4 * 1024**3, 'evaluate': 4 * 1024**3,
                         'predict': 512 * 1024**2, 'batch_predict': 8 * 1024**3}


def enabled():
    return os.environ.get('STORAGE_ENFORCEMENT', '').lower() == 'true'


def owner_key(owner_id):
    if not isinstance(owner_id, str) or not owner_id:
        raise ValueError('Owner is required')
    return hashlib.sha256(owner_id.encode('utf-8')).hexdigest()[:32]


def reserve(owner_id, key, kind, required_bytes):
    if enabled():
        return ledger.reserve(owner_id, key, kind, required_bytes)


def prepare(owner_id, key, namespace, relative_path, *, connection=None):
    if enabled():
        ledger.prepare_directory(owner_id, key, namespace, str(relative_path), connection=connection)


def release(owner_id, key):
    if enabled():
        try:
            return ledger.release(owner_id, key)
        except StorageReservationError as exc:
            if exc.code != 'RESERVATION_NOT_FOUND':
                raise


def check_storage_error(exc):
    if isinstance(exc, OSError) and exc.errno in {errno.EDQUOT, errno.ENOSPC}:
        raise StorageReservationError('STORAGE_QUOTA_EXCEEDED',
            'Storage is full. Remove unused files or ask an administrator to increase your storage limit.') from exc


@contextmanager
def write_lease(owner_id, required_bytes):
    lease = RequestLease(owner_id, required_bytes).open() if enabled() else None
    try:
        yield lease
    except OSError as exc:
        check_storage_error(exc)
        raise
    finally:
        if lease:
            lease.close()


def begin_compute(row):
    if not enabled():
        return
    job_id = str(uuid.UUID(str(row['id'])))
    key = f'compute:{job_id}:execute'
    reserve(row['owner_user_id'], key, row['kind'], _COMPUTE_OUTPUT_BYTES[row['kind']])
    prepare(row['owner_user_id'], key, 'runs', Path('.compute') / job_id)
    if row['kind'] in {'train', 'evaluate'}:
        from settings import RUNS_DIR
        from dataset_storage import registered_storage_path
        with registry._connect() as connection:
            run = connection.execute('select storage_path from training_tasks where id=%s and owner_user_id=%s',
                                     (row['run_id'], row['owner_user_id'])).fetchone()
        if not run:
            raise StorageReservationError('STORAGE_CONFLICT', 'The job has no owned training directory.')
        path = registered_storage_path(RUNS_DIR, run['storage_path'])
        prepare(row['owner_user_id'], key, 'runs', path.relative_to(RUNS_DIR))


def finish_compute(row):
    if enabled():
        release(row['owner_user_id'], f"compute:{uuid.UUID(str(row['id']))}:execute")


def recover_compute_reservations():
    if not enabled():
        return
    with registry._connect() as connection:
        rows = connection.execute('''select j.* from compute_jobs j join storage_reservations r
            on r.owner_user_id=j.owner_user_id and r.operation_key='compute:'||j.id::text||':execute'
            where r.released_at is null and j.status in ('completed','failed','cancelled')''').fetchall()
    for row in rows:
        finish_compute(row)


@contextmanager
def lease_registry():
    root = Path(os.environ.get('STORAGE_LEASE_DIR', '/storage-leases'))
    fd = os.open(root / '.registry.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield root
    finally:
        os.close(fd)


def recover_requests():
    if not enabled():
        return
    root = Path(os.environ.get('STORAGE_LEASE_DIR', '/storage-leases'))
    for path in root.iterdir():
        match = re.fullmatch(r'(backend|label)-([0-9a-f]{32})', path.name)
        if not match:
            continue
        fd = None
        try:
            with lease_registry():
                fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
                if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
                    raise ValueError('Invalid request lease')
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            key = f'http:{match[1]}:{match[2]}'
            with registry._connect() as connection:
                rows = connection.execute('''select * from storage_reservations
                    where operation_key=%s or operation_key like %s order by created_at''', (key, key + ':%')).fetchall()
                owners = {row['owner_user_id'] for row in rows}
                if len(owners) > 1:
                    raise ValueError('Request lease owner is ambiguous')
                account = connection.execute('select * from storage_accounts where owner_user_id=%s',
                                             (next(iter(owners)),)).fetchone() if owners else None
            if account:
                ledger._call('cleanup-upload', account, namespace='storage' if match[1] == 'backend' else 'label-studio',
                             requestId=match[2])
                for row in rows:
                    release(row['owner_user_id'], row['operation_key'])
            path.unlink(missing_ok=True)
        except (BlockingIOError, FileNotFoundError):
            pass
        except Exception:
            log.exception('Request lease retained for recovery: %s', path.name)
        finally:
            if fd is not None:
                os.close(fd)


class RequestLease:
    """An OS lock proves a dead request is no longer writing, unlike a timeout."""

    def __init__(self, owner_id, required_bytes):
        self.owner_id = owner_id
        self.id = uuid.uuid4().hex
        self.key = f'http:backend:{self.id}'
        self.required_bytes = required_bytes
        self.relative_path = Path('.uploads') / owner_key(owner_id) / self.id
        self.directory = STORAGE_DIR / self.relative_path
        self.lease_file = Path(os.environ.get('STORAGE_LEASE_DIR', '/storage-leases')) / f'backend-{self.id}'
        self.fd = None
        self.extra_keys = []

    def open(self):
        if not enabled():
            raise RuntimeError('Request leases require storage enforcement')
        with lease_registry():
            self.fd = os.open(self.lease_file, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            reserve(self.owner_id, self.key, 'upload', self.required_bytes)
            prepare(self.owner_id, self.key, 'storage', self.relative_path.parent)
            prepare(self.owner_id, self.key, 'storage', self.relative_path)
            return self
        except BaseException:
            self.close()
            raise

    def additional(self, purpose, required_bytes, *, namespace=None, path=None):
        key = self.key + ':' + purpose
        if key not in self.extra_keys:
            self.extra_keys.append(key)
        reserve(self.owner_id, key, 'upload', required_bytes)
        if namespace is not None:
            prepare(self.owner_id, key, namespace, path)
        return key

    def close(self):
        if self.fd is None:
            return
        try:
            if self.directory.exists():
                shutil.rmtree(self.directory)
            for key in [*self.extra_keys, self.key]:
                release(self.owner_id, key)
            self.lease_file.unlink(missing_ok=True)
        except Exception:
            # Keep the lock file and charged allowance for confirmed-dead-request recovery.
            log.exception('Request storage cleanup retained for recovery: %s', self.id)
        finally:
            os.close(self.fd)
            self.fd = None
