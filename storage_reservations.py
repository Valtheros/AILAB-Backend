"""Durable write allowances backed by per-owner filesystem project quotas."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import re
from typing import Callable

from api_errors import PublicError
from resource_repository import ResourceRepository
from storage_operator_client import StorageOperatorClient, StorageOperatorError


DEFAULT_LIMIT_BYTES = 20 * 1024**3
DEFAULT_FREE_RESERVE_BYTES = 20 * 1024**3
_BOOTSTRAP_BYTES = 65536
_MAX_BYTES = 2**63 - 1024
_KINDS = frozenset({
    'upload', 'import', 'label_upload', 'label_publish', 'train', 'evaluate',
    'predict', 'batch_predict', 'index', 'transfer',
})
_OPERATION_KEY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$')
_GLOBAL_LOCK = ResourceRepository._lock_key('storage-reservations:global')


class StorageReservationError(PublicError):
    def __init__(self, code: str, message: str):
        super().__init__(code, message, status=503 if code == 'STORAGE_UNAVAILABLE' else 409)


def _valid_bytes(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_BYTES


def _blocks(value: int) -> int:
    return max(1024, ((value + 1023) // 1024) * 1024)


def _operation_key(value):
    if not isinstance(value, str) or not _OPERATION_KEY.fullmatch(value):
        raise ValueError('Invalid storage operation key')


class StorageReservations:
    def __init__(self, connect: Callable, operator=None,
                 *, free_reserve_bytes: int = DEFAULT_FREE_RESERVE_BYTES):
        if not _valid_bytes(free_reserve_bytes):
            raise ValueError('Invalid filesystem reserve')
        self._connect = connect
        self._operator = operator or StorageOperatorClient()
        self._free_reserve_bytes = free_reserve_bytes

    @contextmanager
    def _locked(self, owner_id: str, *, active=True):
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Owner is required')
        # Session locks survive the commit-before-grant required by the quota operator.
        with self._connect() as connection:
            connection.autocommit = True
            connection.execute("set lock_timeout='5s'")
            connection.execute("set statement_timeout='10s'")
            connection.execute('select pg_advisory_lock(%s)',
                               (ResourceRepository._lock_key(f'user-resources:{owner_id}'),))
            connection.execute('select pg_advisory_lock(%s)', (_GLOBAL_LOCK,))
            if active:
                user = connection.execute('select id,coalesce(banned,false) as banned from "user" where id=%s',
                                          (owner_id,)).fetchone()
                if not user or user['banned']:
                    raise StorageReservationError('ACCOUNT_UNAVAILABLE', 'User account is unavailable.')
            yield connection

    def _call(self, operation, account=None, **values):
        try:
            result = self._operator.call(operation, account['project_id'] if account else 0,
                                         account['owner_user_id'] if account else '', **values)
            if any(not _valid_bytes(result.get(key)) for key in ('usedBytes', 'hardLimitBytes', 'freeBytes')):
                raise ValueError('Invalid storage measurement')
            if account and result['hardLimitBytes'] == 0:
                raise ValueError('Unlimited filesystem project')
            return result
        except (StorageOperatorError, OSError, ValueError, KeyError, TypeError) as exc:
            code = exc.code if isinstance(exc, StorageOperatorError) else 'STORAGE_UNAVAILABLE'
            raise StorageReservationError(code, 'Storage accounting is not available.') from exc

    def _measure(self, account):
        result = self._call('measure', account)
        if result['hardLimitBytes'] > account['enforced_limit_bytes']:
            raise StorageReservationError('STORAGE_UNAVAILABLE', 'Filesystem allowance is not accounted for.')
        return result

    def _available(self, connection):
        free = self._call('capacity')['freeBytes']
        outstanding = 0
        for account in connection.execute('select * from storage_accounts').fetchall():
            if account['state'] == 'ready':
                usage = self._measure(account)
                free = min(free, usage['freeBytes'])
                outstanding += max(0, account['enforced_limit_bytes'] - usage['usedBytes'])
            else:
                outstanding += account['enforced_limit_bytes']
        return free - outstanding - self._free_reserve_bytes

    def _account(self, connection, owner_id):
        account = connection.execute('select * from storage_accounts where owner_user_id=%s', (owner_id,)).fetchone()
        if not account:
            if self._available(connection) < _BOOTSTRAP_BYTES:
                raise StorageReservationError('STORAGE_DISK_FULL', 'The server needs free space for system operations.')
            account = connection.execute('insert into storage_accounts(owner_user_id) values(%s) returning *',
                                         (owner_id,)).fetchone()
        if account['state'] == 'pending':
            usage = self._call('provision', account)
            if usage['hardLimitBytes'] > account['enforced_limit_bytes']:
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Existing storage requires verified backfill.')
            self._call('limit', account, limitBytes=account['enforced_limit_bytes'])
            account = connection.execute("update storage_accounts set state='ready',updated_at=now() where owner_user_id=%s returning *",
                                         (owner_id,)).fetchone()
        if account['state'] != 'ready':
            raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting is not ready.')
        return account

    def prepare_account(self, owner_id):
        with self._locked(owner_id) as connection:
            return dict(self._account(connection, owner_id))

    def reserve(self, owner_id: str, operation_key: str, kind: str, requested_bytes: int) -> dict:
        _operation_key(operation_key)
        if not isinstance(kind, str) or kind not in _KINDS or not _valid_bytes(requested_bytes) or requested_bytes == 0:
            raise ValueError('Invalid storage reservation')
        with self._locked(owner_id) as connection:
            account = self._account(connection, owner_id)
            from storage_transfer import pending as transfer_pending
            if transfer_pending(connection, owner_id):
                raise StorageReservationError('STORAGE_IN_USE', 'An account transfer is in progress. Retry when it has finished.')
            existing = connection.execute('select * from storage_reservations where owner_user_id=%s and operation_key=%s',
                                          (owner_id, operation_key)).fetchone()
            if existing:
                if existing['kind'] != kind or existing['reserved_bytes'] != requested_bytes:
                    raise StorageReservationError('RESERVATION_CONFLICT', 'Operation already has a different reservation.')
                if existing['released_at'] is not None:
                    raise StorageReservationError('RESERVATION_CLOSED', 'Operation reservation has already been released.')
                self._measure(account)
                self._call('limit', account, limitBytes=account['enforced_limit_bytes'])
                return dict(existing)
            usage = self._measure(account)
            pending = connection.execute('''select coalesce(sum(reserved_bytes),0) as bytes
                from storage_reservations where owner_user_id=%s and released_at is null''', (owner_id,)).fetchone()['bytes']
            needed = _blocks(usage['usedBytes'] + int(pending) + requested_bytes)
            if needed > account['limit_bytes']:
                raise StorageReservationError('STORAGE_QUOTA_EXCEEDED', 'The user storage quota would be exceeded.')
            available = self._available(connection)
            if max(0, needed - account['enforced_limit_bytes']) > available:
                raise StorageReservationError('STORAGE_DISK_FULL', 'The server needs free space for system operations.')
            # Account for the entire grant, so output is not capped at its estimate.
            ceiling = max(account['enforced_limit_bytes'], min(account['limit_bytes'],
                ((account['enforced_limit_bytes'] + available) // 1024) * 1024))
            # A crash may leave a paid-for allowance, never an unaccounted grant.
            with connection.transaction():
                row = connection.execute('''insert into storage_reservations(owner_user_id,operation_key,kind,reserved_bytes)
                    values(%s,%s,%s,%s) returning *''', (owner_id, operation_key, kind, requested_bytes)).fetchone()
                connection.execute('update storage_accounts set enforced_limit_bytes=%s,updated_at=now() where owner_user_id=%s',
                                   (ceiling, owner_id))
            self._call('limit', account, limitBytes=ceiling)
            return dict(row)

    def prepare_directory(self, owner_id, operation_key, namespace, path, *, connection=None):
        _operation_key(operation_key)
        with (self._locked(owner_id) if connection is None else nullcontext(connection)) as connection:
            connection.execute('select pg_advisory_xact_lock(%s)',
                               (ResourceRepository._lock_key(f'user-resources:{owner_id}'),))
            account = connection.execute('select * from storage_accounts where owner_user_id=%s', (owner_id,)).fetchone()
            if not account or account['state'] != 'ready':
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting is not ready.')
            reservation = connection.execute('''select id from storage_reservations where owner_user_id=%s
                and operation_key=%s and released_at is null''', (owner_id, operation_key)).fetchone()
            if not reservation:
                raise StorageReservationError('RESERVATION_CLOSED', 'A live write reservation is required.')
            self._measure(account)
            self._call('limit', account, limitBytes=account['enforced_limit_bytes'])
            self._call('prepare', account, namespace=namespace, path=path)

    def release(self, owner_id: str, operation_key: str) -> dict:
        """Caller must prove that the operation and its child processes stopped writing."""
        _operation_key(operation_key)
        with self._locked(owner_id, active=False) as connection:
            row = connection.execute('select * from storage_reservations where owner_user_id=%s and operation_key=%s',
                                     (owner_id, operation_key)).fetchone()
            if not row:
                raise StorageReservationError('RESERVATION_NOT_FOUND', 'Storage reservation was not found.')
            if row['released_at'] is not None:
                return dict(row)
            account = connection.execute('select * from storage_accounts where owner_user_id=%s', (owner_id,)).fetchone()
            usage = self._measure(account)
            remaining = connection.execute('''select coalesce(sum(reserved_bytes),0) as bytes from storage_reservations
                where owner_user_id=%s and released_at is null and id<>%s''', (owner_id, row['id'])).fetchone()['bytes']
            # Other operations may already be using shared owner headroom, not just their estimates.
            ceiling = account['enforced_limit_bytes'] if remaining else min(account['enforced_limit_bytes'], _blocks(usage['usedBytes']))
            # Revoke kernel capacity before making the database allowance reusable.
            self._call('limit', account, limitBytes=ceiling)
            with connection.transaction():
                connection.execute('update storage_accounts set enforced_limit_bytes=%s,updated_at=now() where owner_user_id=%s',
                                   (ceiling, owner_id))
                row = connection.execute('update storage_reservations set released_at=now() where id=%s returning *',
                                         (row['id'],)).fetchone()
            return dict(row)

    def usage(self, owner_id):
        with self._locked(owner_id, active=False) as connection:
            account = connection.execute('select * from storage_accounts where owner_user_id=%s', (owner_id,)).fetchone()
            if not account or account['state'] != 'ready':
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting is not ready.')
            used = self._measure(account)['usedBytes']
            pending = connection.execute('''select coalesce(sum(reserved_bytes),0) as bytes from storage_reservations
                where owner_user_id=%s and released_at is null''', (owner_id,)).fetchone()['bytes']
            return {'usedBytes': used, 'limitBytes': account['limit_bytes'], 'reservedBytes': int(pending)}

    def set_limit(self, owner_id, limit_bytes):
        if not _valid_bytes(limit_bytes) or limit_bytes < 1024 or limit_bytes % 1024:
            raise ValueError('Storage limit must be a positive multiple of 1024 bytes')
        with self._locked(owner_id, active=False) as connection:
            account = connection.execute('select * from storage_accounts where owner_user_id=%s', (owner_id,)).fetchone()
            if not account or account['state'] != 'ready':
                raise StorageReservationError('STORAGE_UNAVAILABLE', 'Storage accounting is not ready.')
            ceiling = min(account['enforced_limit_bytes'], limit_bytes)
            self._measure(account)
            self._call('limit', account, limitBytes=ceiling)
            connection.execute('update storage_accounts set limit_bytes=%s,enforced_limit_bytes=%s,updated_at=now() where owner_user_id=%s',
                               (limit_bytes, ceiling, owner_id))

    @staticmethod
    def assert_can_retire(connection, owner_id):
        from storage_transfer import pending
        if pending(connection, owner_id):
            raise StorageReservationError('STORAGE_IN_USE', 'Finish recovering the account transfer before deleting its data.')
        if not connection.execute('select 1 from "user" where id=%s and coalesce(banned,false)', (owner_id,)).fetchone():
            raise StorageReservationError('ACCOUNT_IN_USE', 'Disable this account before deleting its data.')
        if connection.execute('''select 1 from storage_reservations where owner_user_id=%s
                and released_at is null limit 1''', (owner_id,)).fetchone():
            raise StorageReservationError('STORAGE_IN_USE', 'Wait for active uploads and jobs to stop before deleting this account.')

    def retire(self, owner_id):
        with self._locked(owner_id, active=False) as connection:
            account = connection.execute('select * from storage_accounts where owner_user_id=%s', (owner_id,)).fetchone()
            if not account:
                return
            self.assert_can_retire(connection, owner_id)
            self._measure(account)
            self._call('retire', account)
            with connection.transaction():
                connection.execute('delete from storage_reservations where owner_user_id=%s', (owner_id,))
                connection.execute('delete from storage_accounts where owner_user_id=%s', (owner_id,))

    def trim_idle(self, owner_id):
        with self._locked(owner_id, active=False) as connection:
            account = connection.execute("select * from storage_accounts where owner_user_id=%s and state='ready'", (owner_id,)).fetchone()
            if not account or connection.execute('select 1 from storage_reservations where owner_user_id=%s and released_at is null', (owner_id,)).fetchone():
                return
            from storage_transfer import pending
            if pending(connection, owner_id):
                return
            ceiling = min(account['enforced_limit_bytes'], _blocks(self._measure(account)['usedBytes']))
            self._call('limit', account, limitBytes=ceiling)
            connection.execute('update storage_accounts set enforced_limit_bytes=%s,updated_at=now() where owner_user_id=%s', (ceiling, owner_id))
