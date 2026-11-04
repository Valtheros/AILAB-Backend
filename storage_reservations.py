"""Conservative storage reservations; callers must separately enforce project quotas."""

from __future__ import annotations

import re
from typing import Callable

from resource_repository import ResourceRepository


DEFAULT_LIMIT_BYTES = 20 * 1024**3
DEFAULT_FREE_RESERVE_BYTES = 20 * 1024**3
_MAX_BYTES = 2**63 - 1
_KINDS = frozenset({
    'upload', 'import', 'label_upload', 'label_publish', 'train', 'evaluate',
    'predict', 'batch_predict', 'index', 'transfer',
})
_OPERATION_KEY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$')
_GLOBAL_LOCK = ResourceRepository._lock_key('storage-reservations:global')


class StorageReservationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _valid_bytes(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_BYTES


class StorageReservations:
    """Reserve against trusted project usage, hard limit and filesystem-wide free bytes.

    Measurements run while the shared database lock is held, so the provider
    must have a bounded timeout. Reservations remain fully charged until the
    caller confirms that all associated writes have stopped.
    """

    def __init__(self, connect: Callable, measure: Callable[[int], tuple[int, int, int]],
                 *, free_reserve_bytes: int = DEFAULT_FREE_RESERVE_BYTES):
        if not _valid_bytes(free_reserve_bytes):
            raise ValueError('Invalid filesystem reserve')
        self._connect = connect
        self._measure = measure
        self._free_reserve_bytes = free_reserve_bytes

    @staticmethod
    def _lock_owner(connection, owner_id: str):
        connection.execute('select pg_advisory_xact_lock(%s)',
                           (ResourceRepository._lock_key(f'user-resources:{owner_id}'),))

    @staticmethod
    def _require_active_user(connection, owner_id: str):
        user = connection.execute('select id,coalesce(banned,false) as banned from "user" where id=%s for share',
                                  (owner_id,)).fetchone()
        if not user or user['banned']:
            raise StorageReservationError('ACCOUNT_UNAVAILABLE', 'User account is unavailable.')

    def prepare_account(self, owner_id: str) -> dict:
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Owner is required')
        with self._connect() as connection:
            self._lock_owner(connection, owner_id)
            self._require_active_user(connection, owner_id)
            connection.execute('insert into storage_accounts(owner_user_id) values(%s) on conflict(owner_user_id) do nothing',
                               (owner_id,))
            return dict(connection.execute('select * from storage_accounts where owner_user_id=%s',
                                           (owner_id,)).fetchone())

    def reserve(self, owner_id: str, operation_key: str, kind: str, requested_bytes: int) -> dict:
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Owner is required')
        if not isinstance(operation_key, str) or not _OPERATION_KEY.fullmatch(operation_key):
            raise ValueError('Invalid storage operation key')
        if not isinstance(kind, str) or kind not in _KINDS or not _valid_bytes(requested_bytes) or requested_bytes == 0:
            raise ValueError('Invalid storage reservation')
        with self._connect() as connection:
            self._lock_owner(connection, owner_id)
            connection.execute('select pg_advisory_xact_lock(%s)', (_GLOBAL_LOCK,))
            self._require_active_user(connection, owner_id)
            account = connection.execute('select * from storage_accounts where owner_user_id=%s for update',
                                         (owner_id,)).fetchone()
            if not account or account['state'] != 'ready':
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting is not ready.')
            existing = connection.execute('''select * from storage_reservations
                where owner_user_id=%s and operation_key=%s for update''', (owner_id, operation_key)).fetchone()
            if existing:
                if existing['kind'] != kind or existing['reserved_bytes'] != requested_bytes:
                    raise StorageReservationError('RESERVATION_CONFLICT', 'Operation already has a different reservation.')
                if existing['released_at'] is not None:
                    raise StorageReservationError('RESERVATION_CLOSED', 'Operation reservation has already been released.')
                return dict(existing)
            try:
                used, hard_limit, free = self._measure(account['project_id'])
            except Exception as exc:
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting is not available.') from exc
            if not _valid_bytes(used) or not _valid_bytes(hard_limit) or not _valid_bytes(free):
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting returned invalid data.')
            if hard_limit != account['limit_bytes']:
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage limit is not synchronized with the filesystem.')
            pending = connection.execute('''select
                coalesce(sum(reserved_bytes) filter(where owner_user_id=%s),0) as owner_bytes,
                coalesce(sum(reserved_bytes),0) as total_bytes
                from storage_reservations where released_at is null''', (owner_id,)).fetchone()
            if used + pending['owner_bytes'] + requested_bytes > account['limit_bytes']:
                raise StorageReservationError('STORAGE_QUOTA_EXCEEDED', 'The user storage quota would be exceeded.')
            if pending['total_bytes'] + requested_bytes + self._free_reserve_bytes > free:
                raise StorageReservationError('STORAGE_DISK_FULL', 'The server needs free space for system operations.')
            row = connection.execute('''insert into storage_reservations(owner_user_id,operation_key,kind,reserved_bytes)
                values(%s,%s,%s,%s) returning *''', (owner_id, operation_key, kind, requested_bytes)).fetchone()
            return dict(row)

    def release(self, owner_id: str, operation_key: str) -> dict:
        """Release only after the operation and its child processes stop writing."""
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Owner is required')
        if not isinstance(operation_key, str) or not _OPERATION_KEY.fullmatch(operation_key):
            raise ValueError('Invalid storage operation key')
        with self._connect() as connection:
            self._lock_owner(connection, owner_id)
            connection.execute('select pg_advisory_xact_lock(%s)', (_GLOBAL_LOCK,))
            row = connection.execute('''select * from storage_reservations
                where owner_user_id=%s and operation_key=%s for update''', (owner_id, operation_key)).fetchone()
            if not row:
                raise StorageReservationError('RESERVATION_NOT_FOUND', 'Storage reservation was not found.')
            if row['released_at'] is None:
                row = connection.execute('''update storage_reservations set released_at=now()
                    where id=%s returning *''', (row['id'],)).fetchone()
            return dict(row)
