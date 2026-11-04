"""Storage reservation tests must run against a dedicated disposable database."""

from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from resource_repository import resource_repository as registry
from storage_reservations import DEFAULT_LIMIT_BYTES, StorageReservationError, StorageReservations
from tests.isolation import require_test_database, require_test_storage


@unittest.skipUnless(os.getenv('COMPUTE_TESTS') == '1', 'Requires isolated PostgreSQL test stack')
class StorageReservationsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_test_database(registry.database_url)
        require_test_storage('RUNS_DIR', 'DATASET_DIR', 'STORAGE_DIR', 'TMPDIR')
        from scripts.apply_schema import apply
        with registry._connect() as connection:
            connection.execute('''create table if not exists "user"(id text primary key,email text unique,name text,
                "emailVerified" boolean default true,banned boolean default false,role text default 'user')''')
            apply(connection)
            apply(connection)

    def setUp(self):
        with registry._connect() as connection:
            connection.execute('truncate storage_reservations,storage_accounts,"user" cascade')
            connection.execute('''insert into "user"(id,email) values
                ('alice','alice@example.invalid'),('bob','bob@example.invalid')''')
        self.measurements = {'used': 0, 'hard': 1024, 'free': 10000}
        self.ledger = StorageReservations(registry._connect,
            lambda _project_id: (self.measurements['used'], self.measurements['hard'], self.measurements['free']),
            free_reserve_bytes=200)

    def ready(self, owner, limit=1024):
        account = self.ledger.prepare_account(owner)
        with registry._connect() as connection:
            connection.execute("update storage_accounts set state='ready',limit_bytes=%s where owner_user_id=%s",
                               (limit, owner))
        return account

    def test_account_defaults_project_ids_and_pending_state(self):
        alice = self.ledger.prepare_account('alice')
        bob = self.ledger.prepare_account('bob')
        self.assertEqual(alice['limit_bytes'], DEFAULT_LIMIT_BYTES)
        self.assertNotEqual(alice['project_id'], bob['project_id'])
        self.assertEqual(alice['project_id'], self.ledger.prepare_account('alice')['project_id'])
        with self.assertRaises(StorageReservationError) as error:
            self.ledger.reserve('alice', 'upload:1', 'upload', 1)
        self.assertEqual(error.exception.code, 'STORAGE_UNAVAILABLE')

    def test_concurrent_reservations_cannot_exceed_one_users_limit(self):
        self.ready('alice')

        def reserve(key):
            try:
                return self.ledger.reserve('alice', key, 'upload', 600)['id']
            except StorageReservationError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, ('upload:1', 'upload:2')))
        self.assertEqual(sum(value == 'STORAGE_QUOTA_EXCEEDED' for value in results), 1)
        with registry._connect() as connection:
            total = connection.execute('''select coalesce(sum(reserved_bytes),0) as total
                from storage_reservations where released_at is null''').fetchone()['total']
        self.assertEqual(total, 600)

    def test_concurrent_users_cannot_spend_the_system_reserve(self):
        self.ready('alice')
        self.ready('bob')
        self.measurements['free'] = 1000

        def reserve(owner):
            try:
                return self.ledger.reserve(owner, 'upload:1', 'upload', 600)['id']
            except StorageReservationError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, ('alice', 'bob')))
        self.assertEqual(sum(value == 'STORAGE_DISK_FULL' for value in results), 1)

    def test_idempotent_reservation_and_release_remain_owner_scoped(self):
        self.ready('alice')
        self.ready('bob')
        first = self.ledger.reserve('alice', 'upload:1', 'upload', 500)
        self.assertEqual(self.ledger.reserve('alice', 'upload:1', 'upload', 500)['id'], first['id'])
        with self.assertRaises(StorageReservationError) as conflict:
            self.ledger.reserve('alice', 'upload:1', 'upload', 501)
        self.assertEqual(conflict.exception.code, 'RESERVATION_CONFLICT')
        with self.assertRaises(StorageReservationError) as other_owner:
            self.ledger.release('bob', 'upload:1')
        self.assertEqual(other_owner.exception.code, 'RESERVATION_NOT_FOUND')
        released = self.ledger.release('alice', 'upload:1')
        self.assertIsNotNone(released['released_at'])
        self.assertEqual(self.ledger.release('alice', 'upload:1')['released_at'], released['released_at'])
        with self.assertRaises(StorageReservationError) as closed:
            self.ledger.reserve('alice', 'upload:1', 'upload', 500)
        self.assertEqual(closed.exception.code, 'RESERVATION_CLOSED')
        self.ledger.reserve('alice', 'upload:2', 'upload', 1000)

    def test_usage_and_unavailable_measurements_fail_closed(self):
        self.ready('alice')
        self.measurements['used'] = 1000
        with self.assertRaises(StorageReservationError) as error:
            self.ledger.reserve('alice', 'upload:1', 'upload', 25)
        self.assertEqual(error.exception.code, 'STORAGE_QUOTA_EXCEEDED')
        self.measurements['used'] = -1
        with self.assertRaises(StorageReservationError) as error:
            self.ledger.reserve('alice', 'upload:1', 'upload', 1)
        self.assertEqual(error.exception.code, 'STORAGE_UNAVAILABLE')
        unavailable = StorageReservations(registry._connect, lambda _project_id: (_ for _ in ()).throw(OSError()),
                                          free_reserve_bytes=200)
        with self.assertRaises(StorageReservationError) as error:
            unavailable.reserve('alice', 'upload:1', 'upload', 1)
        self.assertEqual(error.exception.code, 'STORAGE_UNAVAILABLE')
        self.measurements['used'] = 0
        for inconsistent in (0, 2048):
            self.measurements['hard'] = inconsistent
            with self.assertRaises(StorageReservationError) as error:
                self.ledger.reserve('alice', 'upload:1', 'upload', 1)
            self.assertEqual(error.exception.code, 'STORAGE_UNAVAILABLE')

    def test_banned_users_cannot_reserve_but_can_release(self):
        self.ready('alice')
        self.ledger.reserve('alice', 'upload:1', 'upload', 500)
        with registry._connect() as connection:
            connection.execute('update "user" set banned=true where id=%s', ('alice',))
        with self.assertRaises(StorageReservationError) as error:
            self.ledger.reserve('alice', 'upload:1', 'upload', 500)
        self.assertEqual(error.exception.code, 'ACCOUNT_UNAVAILABLE')
        self.assertIsNotNone(self.ledger.release('alice', 'upload:1')['released_at'])

    def test_account_cleanup_and_transfer_are_blocked_before_label_studio_deletion(self):
        import main
        from starlette.requests import Request

        self.ready('alice')
        with self.assertRaisesRegex(ValueError, 'Storage quota cleanup'):
            registry.transfer_user_resources('alice', 'bob')
        with self.assertRaisesRegex(ValueError, 'Storage quota cleanup'):
            registry.delete_user_resource_records('alice')
        with self.assertRaisesRegex(ValueError, 'Storage quota cleanup'):
            registry.finalize_user_resource_delete('alice')
        request = Request({'type': 'http', 'method': 'DELETE', 'path': '/', 'headers': []})
        with patch.object(main, '_require_admin'), patch('label_studio_jobs.delete_account_data') as cleanup:
            with self.assertRaises(main.HTTPException) as error:
                main.admin_delete_user_resources('alice', request)
            self.assertEqual(error.exception.status_code, 409)
            cleanup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
