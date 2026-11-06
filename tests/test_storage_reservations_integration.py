"""Storage reservation tests must run against a dedicated disposable database."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import uuid
from unittest.mock import patch

from resource_repository import resource_repository as registry
from storage_operator_client import StorageOperatorError
from storage_reservations import DEFAULT_LIMIT_BYTES, StorageReservationError, StorageReservations
from tests.isolation import require_test_database, require_test_storage


MiB = 1024**2


class FakeQuotaOperator:
    def __init__(self):
        self.projects = {}
        self.free = 100 * MiB
        self.fail = None
        self.after_limit = None
        self.prepared = []
        self.roots = {}

    def call(self, operation, project_id, owner, **values):
        if self.fail == operation:
            raise StorageOperatorError()
        if operation == 'capacity':
            return {'usedBytes': 0, 'hardLimitBytes': 0, 'freeBytes': self.free}
        if operation == 'provision' and project_id not in self.projects:
            self.projects[project_id] = {'owner': owner, 'used': 8192, 'hard': 8192}
            self.free -= 8192
        project = self.projects[project_id]
        if project['owner'] != owner:
            raise StorageOperatorError('STORAGE_CONFLICT')
        if operation == 'limit':
            project['hard'] = values['limitBytes']
            if self.after_limit:
                self.after_limit(project_id, owner, values['limitBytes'])
        if operation == 'retire':
            project['used'] = project['hard'] = 8192
        if operation == 'prepare':
            self.prepared.append((project_id, owner, values))
            if values['namespace'] in self.roots:
                (self.roots[values['namespace']] / values['path']).mkdir(parents=True, exist_ok=True)
        if operation == 'cleanup-upload' and values['namespace'] in self.roots:
            from storage_service import owner_key
            path = self.roots[values['namespace']] / '.uploads' / owner_key(owner) / values['requestId']
            if path.exists():
                shutil.rmtree(path)
        return {'usedBytes': project['used'], 'hardLimitBytes': project['hard'], 'freeBytes': self.free}


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
        self.operator = FakeQuotaOperator()
        self.ledger = StorageReservations(registry._connect, self.operator, free_reserve_bytes=MiB)

    def ready(self, owner, limit=2 * MiB):
        account = self.ledger.prepare_account(owner)
        self.ledger.set_limit(owner, limit)
        return account

    @contextmanager
    def transfer_fixture(self):
        import resource_repository
        import storage_service
        import storage_transfer
        from dataset_storage import owner_dataset_path
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            datasets, runs, storage, leases = [root / name for name in ('dataset', 'runs', 'storage', 'leases')]
            for path in (datasets, runs, storage, leases, root / 'label-studio'):
                path.mkdir()
            (leases / '.registry.lock').touch()
            stack.enter_context(patch.dict(os.environ, {'STORAGE_ENFORCEMENT': 'true', 'COMPUTE_ENABLED': 'true', 'STORAGE_LEASE_DIR': str(leases)}))
            for module, values in ((storage_service, {'STORAGE_DIR': storage, 'ledger': self.ledger}),
                                   (storage_transfer, {'DATASET_DIR': datasets, 'RUNS_DIR': runs, 'STORAGE_DIR': storage, 'ledger': self.ledger}),
                                   (resource_repository, {'DATASET_DIR': datasets})):
                for name, value in values.items():
                    stack.enter_context(patch.object(module, name, value))
            self.operator.roots = {'dataset': datasets, 'runs': runs, 'storage': storage}
            self.ready('alice', 64 * MiB)
            self.ready('bob', 64 * MiB)
            self.ledger.reserve('alice', 'fixture:write', 'upload', 8 * MiB)
            source = owner_dataset_path(datasets, 'alice', 'dataset')
            self.ledger.prepare_directory('alice', 'fixture:write', 'dataset', str(source.parent.relative_to(datasets)))
            self.ledger.prepare_directory('alice', 'fixture:write', 'dataset', str(source.relative_to(datasets)))
            (source / 'image').write_bytes(b'original image')
            destination = owner_dataset_path(datasets, 'bob', 'dataset')
            dataset = registry.upsert_dataset('alice', None, 'dataset', source, {})
            run = runs / 'run'
            self.ledger.prepare_directory('alice', 'fixture:write', 'runs', 'run')
            (run / 'weights.pt').write_bytes(b'unchanged checkpoint')
            with registry._connect() as c:
                run_id = c.execute("""insert into training_tasks(owner_user_id,created_by,task_type,model_type,status,run_slug,storage_path,dataset_id)
                    values('alice','alice','image_classification','resnet','completed','run',%s,%s) returning id""", (str(run), dataset['id'])).fetchone()['id']
                job_id = uuid.uuid4()
                c.execute("insert into compute_jobs(id,owner_user_id,run_id,kind,mode,status,payload) values(%s,'alice',%s,'predict','cpu','completed','{}')", (job_id, run_id))
                endpoint = c.execute("insert into model_endpoints(owner_user_id,run_id,checkpoint_path,checkpoint_sha256) values('alice',%s,'weights.pt','hash') returning id", (run_id,)).fetchone()['id']
                c.execute("insert into model_api_keys(endpoint_id,key_sha256) values(%s,'test-key-hash')", (endpoint,))
            prediction = runs / '.compute' / str(job_id)
            self.ledger.prepare_directory('alice', 'fixture:write', 'runs', str(prediction.relative_to(runs)))
            (prediction / 'result.json').write_text('{}')
            self.ledger.release('alice', 'fixture:write')
            with registry._connect() as c:
                c.execute('update "user" set banned=true where id=%s', ('alice',))
            yield storage_transfer, source, destination, run, prediction

    def test_transfer_copies_verified_data_to_target_quota_and_revokes_keys(self):
        with self.transfer_fixture() as (_transfer, source, destination, run, prediction):
            old_inode = (run / 'weights.pt').stat().st_ino
            result = registry.transfer_user_resources('alice', 'bob')
            self.assertEqual(result['datasets'], 1)
            self.assertFalse(source.exists())
            self.assertEqual((destination / 'image').read_bytes(), b'original image')
            self.assertEqual((run / 'weights.pt').read_bytes(), b'unchanged checkpoint')
            self.assertNotEqual((run / 'weights.pt').stat().st_ino, old_inode)
            self.assertEqual((prediction / 'result.json').read_text(), '{}')
            with registry._connect() as c:
                self.assertEqual(c.execute('select owner_user_id from datasets').fetchone()['owner_user_id'], 'bob')
                self.assertEqual(c.execute('select owner_user_id from compute_jobs').fetchone()['owner_user_id'], 'bob')
                self.assertFalse(c.execute('select enabled from model_endpoints').fetchone()['enabled'])
                self.assertIsNotNone(c.execute('select revoked_at from model_api_keys').fetchone()['revoked_at'])
                self.assertEqual(c.execute('select state from storage_transfers').fetchone()['state'], 'completed')
            self.assertEqual(self.ledger.usage('bob')['reservedBytes'], 0)

    def test_failed_transfer_restores_original_files_and_database_owner(self):
        with self.transfer_fixture() as (transfer, source, destination, run, _prediction):
            inode = run.stat().st_ino
            with patch.object(transfer.Transfer, 'committed', side_effect=RuntimeError('transaction failed')):
                with self.assertRaisesRegex(RuntimeError, 'transaction failed'):
                    registry.transfer_user_resources('alice', 'bob')
            self.assertEqual((source / 'image').read_bytes(), b'original image')
            self.assertFalse(destination.exists())
            self.assertEqual(run.stat().st_ino, inode)
            with registry._connect() as c:
                self.assertEqual(c.execute('select owner_user_id from training_tasks').fetchone()['owner_user_id'], 'alice')
                self.assertEqual(c.execute('select state from storage_transfers').fetchone()['state'], 'rolled_back')
                self.assertTrue(c.execute('select enabled from model_endpoints').fetchone()['enabled'])

    def test_interrupted_transfer_is_recovered_before_new_writes(self):
        with self.transfer_fixture() as (transfer, source, destination, run, _prediction):
            inode = run.stat().st_ino
            with patch.object(transfer.Transfer, 'committed', side_effect=RuntimeError('lost transaction')), \
                 patch.object(transfer, 'finish', side_effect=RuntimeError('service stopped')):
                with self.assertRaisesRegex(RuntimeError, 'service stopped'):
                    registry.transfer_user_resources('alice', 'bob')
            with self.assertRaises(StorageReservationError) as busy:
                self.ledger.reserve('bob', 'next-upload', 'upload', 1024)
            self.assertEqual(busy.exception.code, 'STORAGE_IN_USE')
            transfer.recover()
            transfer.recover()
            self.assertEqual(run.stat().st_ino, inode)
            self.assertEqual((source / 'image').read_bytes(), b'original image')
            self.assertFalse(destination.exists())
            self.ledger.reserve('bob', 'next-upload', 'upload', 1024)

    def test_transfer_quota_failure_preserves_source_and_never_creates_destination(self):
        with self.transfer_fixture() as (_transfer, source, destination, _run, _prediction):
            self.ledger.set_limit('bob', MiB)
            with self.assertRaises(StorageReservationError) as full:
                registry.transfer_user_resources('alice', 'bob')
            self.assertEqual(full.exception.code, 'STORAGE_QUOTA_EXCEEDED')
            self.assertEqual((source / 'image').read_bytes(), b'original image')
            self.assertFalse(destination.exists())

    @unittest.skipUnless(os.getenv('AILAB_REAL_QUOTA_RUNTIME') == '1', 'requires disposable ext4 runtime')
    def test_transfer_on_real_ext4_assigns_every_copy_and_reclaims_source_usage(self):
        from storage_operator import QuotaOperator
        from storage_project_quota import directory_project
        class RealOperator:
            def call(self, operation, project_id, owner, **values):
                if not hasattr(self, 'operator'):
                    root = self.roots['dataset'].parent
                    if root.stat().st_dev != Path('/dev/loop0').stat().st_rdev:
                        raise RuntimeError('Only the disposable loop device may be used')
                    self.operator = QuotaOperator(root, Path('/dev/loop0'))
                return self.operator.dispatch({'operation': operation, 'projectId': project_id, 'owner': owner, **values})
        self.operator = RealOperator()
        self.ledger = StorageReservations(registry._connect, self.operator, free_reserve_bytes=MiB)
        with self.transfer_fixture() as (_transfer, source, destination, run, prediction):
            from scripts.storage_backfill import apply_plan
            from storage_inventory import scan
            from storage_project_quota import _attributes
            root = self.operator.roots['dataset'].parent
            roots = {source.relative_to(root).as_posix(): 'alice',
                     run.relative_to(root).as_posix(): 'alice',
                     prediction.relative_to(root).as_posix(): 'alice'}
            plan = {**scan(root, roots, attributes=_attributes), 'owners': ['alice', 'bob'], 'roots': roots}
            self.assertFalse(plan['issues'])
            with registry._connect() as c:
                c.autocommit = True
                apply_plan(c, root, Path('/dev/loop0'), plan, root / 'backfill-journal.json')
                apply_plan(c, root, Path('/dev/loop0'), plan, root / 'backfill-journal.json')
            before = self.ledger.usage('alice')['usedBytes']
            registry.transfer_user_resources('alice', 'bob')
            target = self.account('bob')['project_id']
            for path in (destination, run, prediction):
                self.assertEqual(directory_project(path), (target, True))
            self.assertFalse(source.exists())
            self.assertLess(self.ledger.usage('alice')['usedBytes'], before)
            self.assertEqual(self.ledger.usage('bob')['reservedBytes'], 0)

    def test_retire_requires_disabled_user_and_no_live_writer_and_recovers_after_failure(self):
        alice = self.ready('alice')
        self.ready('bob')
        with self.assertRaises(StorageReservationError):
            self.ledger.retire('alice')
        self.ledger.reserve('alice', 'test:write', 'upload', MiB)
        with registry._connect() as connection:
            connection.execute('update "user" set banned=true where id=%s', ('alice',))
        with self.assertRaises(StorageReservationError) as busy:
            self.ledger.retire('alice')
        self.assertEqual(busy.exception.code, 'STORAGE_IN_USE')
        self.ledger.release('alice', 'test:write')
        self.operator.fail = 'retire'
        with self.assertRaises(StorageReservationError):
            self.ledger.retire('alice')
        self.assertEqual(self.account('alice')['project_id'], alice['project_id'])
        self.operator.fail = None
        self.ledger.retire('alice')
        self.ledger.retire('alice')
        with registry._connect() as connection:
            self.assertIsNone(connection.execute('select 1 from storage_accounts where owner_user_id=%s', ('alice',)).fetchone())
            self.assertIsNotNone(connection.execute('select 1 from storage_accounts where owner_user_id=%s', ('bob',)).fetchone())
            connection.execute('delete from "user" where id=%s', ('alice',))

    def test_storage_api_is_owner_scoped_and_admin_limit_requires_admin(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient
        from api_errors import PublicError
        import storage_api
        app = FastAPI()
        app.include_router(storage_api.router)
        @app.exception_handler(PublicError)
        async def handle(_request, exc):
            return JSONResponse({'code': exc.code}, status_code=exc.status)
        with patch.dict(os.environ, {'STORAGE_ENFORCEMENT': 'true'}), patch.object(storage_api, 'ledger', self.ledger), TestClient(app) as client:
            self.ready('alice')
            self.ready('bob')
            response = client.get('/api/storage', headers={'x-user-id': 'alice'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['limitBytes'], 2 * MiB)
            self.assertEqual(client.get('/api/storage').status_code, 401)
            self.assertEqual(client.get('/api/storage/admin', headers={'x-user-id': 'alice'}).status_code, 403)
            self.assertEqual(client.put('/api/storage/admin/bob', headers={'x-user-id': 'alice'}, json={'limitGiB': 5}).status_code, 403)
            with registry._connect() as connection:
                connection.execute('update "user" set role=%s where id=%s', ('admin', 'alice'))
            self.assertEqual(client.get('/api/storage/admin', headers={'x-user-id': 'alice'}).json()['total'], 2)
            self.assertEqual(client.put('/api/storage/admin/bob', headers={'x-user-id': 'alice'}, json={'limitGiB': 5}).status_code, 200)
            self.assertEqual(self.account('bob')['limit_bytes'], 5 * 1024**3)
            self.assertEqual(client.put('/api/storage/admin/bob', headers={'x-user-id': 'alice'}, json={'limitGiB': True}).status_code, 422)
            with registry._connect() as connection:
                connection.execute('update "user" set banned=true where id=%s', ('bob',))
            self.assertEqual(client.get('/api/storage', headers={'x-user-id': 'bob'}).status_code, 403)

    def test_label_storage_bridge_requires_secret_live_lock_and_owned_project(self):
        import fcntl
        import uuid
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import label_studio_api
        import storage_service
        app = FastAPI()
        app.include_router(label_studio_api.internal_router)
        @app.middleware('http')
        async def request_id(request, call_next):
            request.state.request_id = 'test-storage'
            return await call_next(request)
        with registry._connect() as connection:
            connection.execute("insert into label_studio_accounts(owner_user_id,ls_user_id) values('alice',1),('bob',2)")
        request_id, external = uuid.uuid4(), uuid.uuid4()
        with tempfile.TemporaryDirectory() as temporary:
            lease = Path(temporary) / ('label-' + request_id.hex)
            lease.touch()
            with patch.dict(os.environ, {'STORAGE_ENFORCEMENT': 'true', 'STORAGE_LEASE_DIR': temporary,
                                         'LABEL_STUDIO_BRIDGE_SECRET': 'test-secret'}), \
                 patch.object(storage_service, 'ledger', self.ledger), patch.object(label_studio_api, 'ledger', self.ledger), TestClient(app) as client:
                url = f'/internal/label-studio/storage/{request_id}/reserve'
                payload = {'project': 1, 'externalId': str(external), 'name': 'Owned', 'requiredBytes': MiB}
                headers = {'x-user-id': 'alice', 'authorization': 'Bearer test-secret'}
                self.assertEqual(client.post(url, json=payload, headers={'x-user-id': 'alice'}).status_code, 401)
                self.assertEqual(client.post(url, json=payload, headers=headers).status_code, 503)
                with lease.open('r+') as held:
                    fcntl.flock(held, fcntl.LOCK_EX)
                    self.assertEqual(client.post(url, json=payload, headers=headers).status_code, 200)
                    self.assertEqual(client.post(url, json=payload, headers={**headers, 'x-user-id': 'bob'}).status_code, 404)
                    self.assertEqual(client.post(url, json={**payload, 'requiredBytes': True}, headers=headers).status_code, 422)
                    response = client.post(url.removesuffix('reserve') + 'release', json={}, headers=headers)
                    self.assertEqual(response.status_code, 200)
                self.assertEqual(self.ledger.usage('alice')['reservedBytes'], 0)
                self.assertTrue(all(row[1] == 'alice' for row in self.operator.prepared))

    def account(self, owner):
        with registry._connect() as connection:
            return dict(connection.execute('select * from storage_accounts where owner_user_id=%s', (owner,)).fetchone())

    def test_account_defaults_owner_binding_and_pending_provision_recovery(self):
        self.operator.fail = 'provision'
        with self.assertRaises(StorageReservationError):
            self.ledger.prepare_account('alice')
        pending = self.account('alice')
        self.assertEqual(pending['state'], 'pending')
        self.operator.fail = None
        alice = self.ledger.prepare_account('alice')
        bob = self.ledger.prepare_account('bob')
        self.assertEqual(alice['state'], 'ready')
        self.assertEqual(alice['limit_bytes'], DEFAULT_LIMIT_BYTES)
        self.assertEqual(alice['project_id'], pending['project_id'])
        self.assertNotEqual(alice['project_id'], bob['project_id'])
        self.assertEqual(self.ledger.prepare_account('alice')['project_id'], alice['project_id'])

    def test_concurrent_reservations_cannot_exceed_one_users_limit(self):
        self.ready('alice')

        def reserve(key):
            try:
                return self.ledger.reserve('alice', key, 'upload', 1280 * 1024)['id']
            except StorageReservationError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, ('upload:1', 'upload:2')))
        self.assertEqual(sum(value == 'STORAGE_QUOTA_EXCEEDED' for value in results), 1)
        with registry._connect() as connection:
            total = connection.execute('select sum(reserved_bytes) as total from storage_reservations where released_at is null').fetchone()['total']
        self.assertEqual(total, 1280 * 1024)

    def test_concurrent_users_cannot_spend_the_system_reserve(self):
        self.ready('alice', 10 * MiB)
        self.ready('bob', 10 * MiB)
        self.operator.free = 4 * MiB

        def reserve(owner):
            try:
                return self.ledger.reserve(owner, 'upload:1', 'upload', 2 * MiB)['id']
            except StorageReservationError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, ('alice', 'bob')))
        self.assertEqual(sum(value == 'STORAGE_DISK_FULL' for value in results), 1)

    def test_idempotent_reservation_release_and_directory_are_owner_scoped(self):
        self.ready('alice')
        self.ready('bob')
        first = self.ledger.reserve('alice', 'upload:1', 'upload', MiB)
        self.assertEqual(self.ledger.reserve('alice', 'upload:1', 'upload', MiB)['id'], first['id'])
        with self.assertRaises(StorageReservationError) as conflict:
            self.ledger.reserve('alice', 'upload:1', 'upload', MiB + 1)
        self.assertEqual(conflict.exception.code, 'RESERVATION_CONFLICT')
        with self.assertRaises(StorageReservationError):
            self.ledger.prepare_directory('bob', 'upload:1', 'dataset', '.owners/alice')
        self.ledger.prepare_directory('alice', 'upload:1', 'dataset', '.owners/alice')
        self.assertEqual(self.operator.prepared[-1][1], 'alice')
        with self.assertRaises(StorageReservationError) as other_owner:
            self.ledger.release('bob', 'upload:1')
        self.assertEqual(other_owner.exception.code, 'RESERVATION_NOT_FOUND')
        released = self.ledger.release('alice', 'upload:1')
        self.assertIsNotNone(released['released_at'])
        self.assertEqual(self.ledger.release('alice', 'upload:1')['released_at'], released['released_at'])
        with self.assertRaises(StorageReservationError) as closed:
            self.ledger.reserve('alice', 'upload:1', 'upload', MiB)
        self.assertEqual(closed.exception.code, 'RESERVATION_CLOSED')
        with self.assertRaises(StorageReservationError):
            self.ledger.prepare_directory('alice', 'upload:1', 'dataset', '.owners/alice')

    def test_grant_is_committed_before_kernel_change_and_timeout_is_retryable(self):
        self.ready('alice')

        def timeout_after_grant(project_id, owner, hard):
            self.assertEqual(self.account(owner)['enforced_limit_bytes'], hard)
            with registry._connect() as connection:
                row = connection.execute('select * from storage_reservations where owner_user_id=%s and released_at is null', (owner,)).fetchone()
            self.assertEqual(row['reserved_bytes'], MiB)
            raise StorageOperatorError()

        self.operator.after_limit = timeout_after_grant
        with self.assertRaises(StorageReservationError):
            self.ledger.reserve('alice', 'train:1', 'train', MiB)
        self.operator.after_limit = None
        retried = self.ledger.reserve('alice', 'train:1', 'train', MiB)
        self.assertIsNone(retried['released_at'])
        self.assertEqual(self.ledger.usage('alice')['reservedBytes'], MiB)

    def test_revocation_happens_before_allowance_is_reusable(self):
        account = self.ready('alice')
        self.ledger.reserve('alice', 'train:1', 'train', MiB)
        old_ceiling = self.account('alice')['enforced_limit_bytes']

        def timeout_after_revoke(project_id, owner, hard):
            self.assertLess(hard, old_ceiling)
            self.assertEqual(self.account(owner)['enforced_limit_bytes'], old_ceiling)
            raise StorageOperatorError()

        self.operator.after_limit = timeout_after_revoke
        with self.assertRaises(StorageReservationError):
            self.ledger.release('alice', 'train:1')
        self.operator.after_limit = None
        self.assertEqual(self.ledger.usage('alice')['reservedBytes'], MiB)
        self.assertLess(self.operator.projects[account['project_id']]['hard'], old_ceiling)
        self.ledger.release('alice', 'train:1')
        self.assertEqual(self.ledger.usage('alice')['reservedBytes'], 0)

    def test_output_can_exceed_estimate_without_exceeding_account_or_global_grant(self):
        account = self.ready('alice', 10 * MiB)
        self.ledger.reserve('alice', 'train:1', 'train', MiB)
        self.assertEqual(self.operator.projects[account['project_id']]['hard'], 10 * MiB)
        self.ledger.reserve('alice', 'upload:1', 'upload', MiB)
        self.operator.projects[account['project_id']]['used'] = 3 * MiB
        self.ledger.release('alice', 'upload:1')
        self.assertEqual(self.operator.projects[account['project_id']]['hard'], 10 * MiB)
        self.ledger.release('alice', 'train:1')
        self.assertEqual(self.operator.projects[account['project_id']]['hard'], 3 * MiB)

    def test_usage_and_unavailable_measurements_fail_closed(self):
        account = self.ready('alice')
        project = self.operator.projects[account['project_id']]
        project['used'] = 2 * MiB
        with self.assertRaises(StorageReservationError) as error:
            self.ledger.reserve('alice', 'upload:1', 'upload', 1024)
        self.assertEqual(error.exception.code, 'STORAGE_QUOTA_EXCEEDED')
        for invalid in (-1, True):
            project['used'] = invalid
            with self.assertRaises(StorageReservationError):
                self.ledger.reserve('alice', 'upload:1', 'upload', 1024)
        project['used'] = 8192
        for hard in (0, 4 * MiB):
            project['hard'] = hard
            with self.assertRaises(StorageReservationError) as error:
                self.ledger.reserve('alice', 'upload:1', 'upload', 1024)
            self.assertEqual(error.exception.code, 'STORAGE_UNAVAILABLE')

    def test_admin_lowering_quota_preserves_usage_and_blocks_new_writes(self):
        account = self.ready('alice')
        self.ledger.reserve('alice', 'train:1', 'train', MiB)
        self.operator.projects[account['project_id']]['used'] = MiB
        self.ledger.set_limit('alice', 512 * 1024)
        self.assertEqual(self.ledger.usage('alice')['usedBytes'], MiB)
        self.assertEqual(self.operator.projects[account['project_id']]['hard'], 512 * 1024)
        with self.assertRaises(StorageReservationError):
            self.ledger.reserve('alice', 'upload:1', 'upload', 1024)
        self.ledger.release('alice', 'train:1')
        self.assertEqual(self.operator.projects[account['project_id']]['hard'], 512 * 1024)

    def test_banned_users_cannot_reserve_but_can_release(self):
        self.ready('alice')
        self.ledger.reserve('alice', 'upload:1', 'upload', MiB)
        with registry._connect() as connection:
            connection.execute('update "user" set banned=true where id=%s', ('alice',))
        with self.assertRaises(StorageReservationError) as error:
            self.ledger.reserve('alice', 'upload:1', 'upload', MiB)
        self.assertEqual(error.exception.code, 'ACCOUNT_UNAVAILABLE')
        self.assertIsNotNone(self.ledger.release('alice', 'upload:1')['released_at'])

    def test_request_recovery_waits_for_real_process_exit_and_cleans_all_allowances(self):
        import storage_service as storage

        self.ready('alice', 10 * MiB)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            leases = root / 'leases'
            leases.mkdir()
            (leases / '.registry.lock').touch()
            self.operator.roots['storage'] = root / 'storage'
            self.operator.roots['dataset'] = root / 'dataset'
            with patch.object(storage, 'ledger', self.ledger), patch.object(storage, 'STORAGE_DIR', root / 'storage'), \
                 patch.dict(os.environ, {'STORAGE_ENFORCEMENT': 'true', 'STORAGE_LEASE_DIR': str(leases)}):
                request = storage.RequestLease('alice', MiB).open()
                request.additional('extract', MiB, namespace='dataset', path='.owners/alice')
                (request.directory / 'spool').write_bytes(b'request bytes')
                storage.recover_requests()
                self.assertTrue(request.lease_file.exists())
                self.assertEqual(self.ledger.usage('alice')['reservedBytes'], 2 * MiB)
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], pass_fds=[request.fd])
                try:
                    os.close(request.fd)
                    request.fd = None
                    storage.recover_requests()
                    self.assertTrue(request.lease_file.exists())
                finally:
                    child.kill()
                    child.wait(timeout=5)
                storage.recover_requests()
                self.assertFalse(request.directory.exists())
                self.assertFalse(request.lease_file.exists())
                self.assertEqual(self.ledger.usage('alice')['reservedBytes'], 0)
                storage.recover_requests()

    def test_request_release_failure_retains_a_recoverable_lock_file(self):
        import storage_service as storage

        self.ready('alice')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / '.registry.lock').touch()
            self.operator.roots['storage'] = root / 'storage'
            with patch.object(storage, 'ledger', self.ledger), patch.object(storage, 'STORAGE_DIR', root / 'storage'), \
                 patch.dict(os.environ, {'STORAGE_ENFORCEMENT': 'true', 'STORAGE_LEASE_DIR': str(root)}):
                request = storage.RequestLease('alice', MiB).open()
                self.operator.fail = 'limit'
                with self.assertLogs('storage_service', level='ERROR'):
                    request.close()
                self.assertTrue(request.lease_file.exists())
                self.assertIsNone(request.fd)
                self.operator.fail = None
                storage.recover_requests()
                self.assertFalse(request.lease_file.exists())
                self.assertEqual(self.ledger.usage('alice')['reservedBytes'], 0)

    def test_account_cleanup_remains_fail_closed_until_lifecycle_is_integrated(self):
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
