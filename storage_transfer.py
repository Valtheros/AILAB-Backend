"""Quota-bound copies with a durable commit/rollback record for account transfers."""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import uuid

from dataset_storage import owner_dataset_path, registered_storage_path
from resource_repository import resource_repository as registry
from settings import DATASET_DIR, RUNS_DIR, STORAGE_DIR
from storage_service import RequestLease, check_storage_error, ledger, lease_registry, prepare
from storage_reservations import StorageReservationError


log = logging.getLogger(__name__)


def pending(connection, owner):
    return connection.execute('''select 1 from storage_transfers where state in ('copying','committed')
        and (source_user_id=%s or target_user_id=%s) limit 1''', (owner, owner)).fetchone()


def tree(path, *, sync=False):
    """Reject links and special files, and include names plus content in the copy check."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Transfer source is not a regular directory')
    device, total, digest = path.stat().st_dev, 0, hashlib.sha256()
    for current, dirs, files in os.walk(path, followlinks=False):
        dirs.sort()
        files.sort()
        for name in ['.', *files, *dirs]:
            item = Path(current) if name == '.' else Path(current) / name
            info = item.lstat()
            if info.st_dev != device or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError('Transfer refuses links, special files and filesystem boundaries')
            if name in dirs:
                continue
            digest.update(json.dumps([item.relative_to(path).as_posix(), stat.S_ISDIR(info.st_mode),
                                      info.st_size if stat.S_ISREG(info.st_mode) else 0]).encode())
            total += max(4096, ((info.st_size + 4095) // 4096) * 4096)
            if stat.S_ISREG(info.st_mode):
                fd = os.open(item, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    before = os.fstat(fd)
                    if (before.st_ino, before.st_dev) != (info.st_ino, info.st_dev):
                        raise ValueError('Source changed during transfer')
                    while chunk := os.read(fd, 1024**2):
                        digest.update(chunk)
                    after = os.fstat(fd)
                    if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
                        raise ValueError('Source changed during transfer')
                    if sync:
                        os.fsync(fd)
                finally:
                    os.close(fd)
            elif sync:
                sync_directory(item)
    return total, digest.hexdigest()


def checked_path(value):
    path = Path(value)
    for root in (DATASET_DIR, RUNS_DIR, STORAGE_DIR):
        if path != root and path.is_relative_to(root):
            resolved = registered_storage_path(root, str(path))
            if resolved != path or path.is_symlink():
                break
            return path
    raise ValueError('Invalid transfer journal path')


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remove_copy(path, inode):
    if path.exists():
        if path.is_symlink() or path.stat().st_ino != inode:
            raise ValueError('Transfer copy was replaced; refusing deletion')
        shutil.rmtree(path)
        sync_directory(path.parent)


def recover_files(plan, committed):
    for item in reversed(plan):
        source, stage, destination = (checked_path(item[key]) for key in ('source', 'stage', 'destination'))
        backup = checked_path(item['backup']) if item['backup'] else None
        if committed:
            original = backup or source
            remove_copy(original, item['sourceInode'])
            remove_copy(stage, item['stageInode'])
        else:
            if backup and backup.exists():
                if backup.stat().st_ino != item['sourceInode']:
                    raise ValueError('Transfer backup was replaced')
                remove_copy(destination, item['stageInode'])
                backup.replace(source)
                sync_directory(source.parent)
            elif not backup:
                remove_copy(destination, item['stageInode'])
            remove_copy(stage, item['stageInode'])


def finish(operation_id):
    with registry._connect() as connection:
        row = connection.execute('select * from storage_transfers where id=%s', (operation_id,)).fetchone()
        if not row or row['state'] not in {'copying', 'committed'}:
            return
        for owner in sorted((row['source_user_id'], row['target_user_id'])):
            connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + owner),))
        row = connection.execute('select * from storage_transfers where id=%s for update', (operation_id,)).fetchone()
        if row['state'] not in {'copying', 'committed'}:
            return
        committed = row['state'] == 'committed'
        recover_files(row['plan'], committed)
        connection.execute('update storage_transfers set state=%s,updated_at=now() where id=%s',
                           ('completed' if committed else 'rolled_back', operation_id))
    for owner in (row['source_user_id'], row['target_user_id']):
        ledger.trim_idle(owner)


def recover():
    with registry._connect() as connection:
        rows = connection.execute("select id,lease_id from storage_transfers where state in ('copying','committed')").fetchall()
    for row in rows:
        fd = None
        try:
            with lease_registry() as root:
                path = root / ('backend-' + row['lease_id'])
                try:
                    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
                except FileNotFoundError:
                    pass
                if fd is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finish(row['id'])
        except BlockingIOError:
            continue
        except Exception:
            log.exception('Storage transfer %s retained for recovery', row['id'])
        finally:
            if fd is not None:
                os.close(fd)


class Transfer:
    def __init__(self, source, target):
        self.source, self.target = source, target
        self.id = uuid.uuid4()
        self.plan = []
        self.lease = None

    def resources(self, connection):
        if not connection.execute('select 1 from "user" where id=%s and coalesce(banned,false)', (self.source,)).fetchone():
            raise ValueError('Disable the source account before transferring its data.')
        pairs = []
        for row in connection.execute("select slug,storage_path from datasets where owner_user_id=%s and status<>'deleted'", (self.source,)):
            pairs.append((registered_storage_path(DATASET_DIR, row['storage_path']), owner_dataset_path(DATASET_DIR, self.target, row['slug'])))
        for row in connection.execute('select storage_path from training_tasks where owner_user_id=%s', (self.source,)):
            if row['storage_path']:
                path = registered_storage_path(RUNS_DIR, row['storage_path'])
                if path.exists():
                    pairs.append((path, path))
        for row in connection.execute('select id from compute_jobs where owner_user_id=%s', (self.source,)):
            path = RUNS_DIR / '.compute' / str(row['id'])
            if path.exists():
                pairs.append((path, path))
        return pairs

    def execute(self):
        with registry._connect() as connection:
            for owner in (self.source, self.target):
                if pending(connection, owner):
                    raise StorageReservationError('STORAGE_IN_USE', 'An account transfer is still being recovered. Retry shortly.')
            pairs = self.resources(connection)
        required = sum(tree(source)[0] for source, _destination in pairs) + 32 * 1024**2
        self.lease = RequestLease(self.target, required).open()
        try:
            return registry.transfer_user_resources(self.source, self.target, _storage_transfer=self)
        except OSError as exc:
            check_storage_error(exc)
            raise
        finally:
            try:
                finish(self.id)
            finally:
                self.lease.close()

    def copy(self, connection):
        for owner in (self.source, self.target):
            if pending(connection, owner):
                raise StorageReservationError('STORAGE_IN_USE', 'Another account transfer is in progress.')
        ledger.assert_can_retire(connection, self.source)
        pairs = self.resources(connection)
        manifests = [(source, destination, *tree(source)) for source, destination in pairs]
        if sum(size for _s, _d, size, _hash in manifests) + 32 * 1024**2 > self.lease.required_bytes:
            raise ValueError('Source data changed before the transfer; retry.')
        destination_parent = owner_dataset_path(DATASET_DIR, self.target, 'pending').parent
        prepare(self.target, self.lease.key, 'dataset', destination_parent.relative_to(DATASET_DIR), connection=connection)
        for index, (source, destination, _size, digest) in enumerate(manifests):
            if source != destination and destination.exists():
                raise FileExistsError('Transfer destination already exists')
            stage = self.lease.directory / str(index)
            stage.mkdir()
            backup = source.with_name('.transfer-' + self.id.hex + '-' + source.name) if source == destination else None
            if backup and backup.exists():
                raise FileExistsError('Transfer backup already exists')
            self.plan.append({'source': str(source), 'destination': str(destination), 'stage': str(stage),
                'sourceInode': source.stat().st_ino, 'stageInode': stage.stat().st_ino,
                'backup': str(backup) if backup else None, 'sha256': digest})
        with registry._connect() as journal:
            journal.execute('''insert into storage_transfers(id,source_user_id,target_user_id,lease_id,state,plan)
                values(%s,%s,%s,%s,'copying',%s::jsonb)''',
                (self.id, self.source, self.target, self.lease.id, json.dumps(self.plan)))
        for item in self.plan:
            source, stage = Path(item['source']), Path(item['stage'])
            shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=False)
            if tree(source)[1] != item['sha256'] or tree(stage, sync=True)[1] != item['sha256']:
                raise ValueError('Copied data failed transfer verification')
        for item in self.plan:
            if item['backup']:
                Path(item['source']).replace(item['backup'])
            Path(item['stage']).replace(item['destination'])
            sync_directory(Path(item['source']).parent)
            sync_directory(Path(item['destination']).parent)
            sync_directory(Path(item['stage']).parent)

    def committed(self, connection):
        connection.execute("update storage_transfers set state='committed',updated_at=now() where id=%s", (self.id,))
