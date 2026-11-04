"""Run only against the dedicated, disposable compute-test database."""
import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from types import SimpleNamespace

import compute_repository as compute
from api_errors import PublicError


@unittest.skipUnless(os.getenv('COMPUTE_TESTS') == '1', 'Requires isolated PostgreSQL test stack')
class ComputeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'ailab_compute_test_pg' not in compute.registry.database_url:
            raise RuntimeError('Refusing to run destructive fixtures against a non-test database')
        from scripts.apply_schema import apply
        with compute.registry._connect() as c:
            c.execute('''create table if not exists "user"(id text primary key,email text unique,name text,
                       "emailVerified" boolean default true,banned boolean default false,role text default 'user')''')
            apply(c)
            apply(c)

    def setUp(self):
        memory = patch('compute_memory.psutil.virtual_memory', return_value=SimpleNamespace(available=21000 * 1048576))
        memory.start()
        self.addCleanup(memory.stop)
        with compute.registry._connect() as c:
            c.execute('truncate compute_jobs,compute_nodes,training_tasks,"user" cascade')
            c.execute('insert into "user"(id,email) values(\'alice\',\'alice@test.invalid\'),(\'bob\',\'bob@test.invalid\')')
            c.execute("insert into compute_nodes(id,ram_total_mb,ram_available_mb,cpu_count) values('local',24576,21000,24)")
            for i, ram in enumerate((8192, 24576)):
                c.execute('''insert into gpu_devices(uuid,node_id,display_index,name,total_mb,free_mb,telemetry_ok,
                    worker_id,worker_heartbeat) values(%s,'local',%s,'Fake GPU',%s,%s,true,%s,now())''', (f'GPU-{i}', i, ram, ram, f'worker-{i}'))
            self.runs = {}
            for owner in ('alice','bob'):
                self.runs[owner] = c.execute('''insert into training_tasks(owner_user_id,created_by,task_type,model_type,status)
                      values(%s,%s,'image_classification','resnet','completed') returning id''', (owner, owner)).fetchone()['id']

    def create(self, owner='alice', kind='predict', **options):
        return compute.create_job(owner, self.runs[owner], kind, {}, {'mode':'auto'}, estimated_vram_mb=2000, **options)

    def test_idempotency_and_cross_user_404(self):
        one = self.create(idempotency_key='same')
        two = self.create(idempotency_key='same')
        self.assertEqual(one['id'], two['id'])
        for fn in (compute.get_job, compute.cancel_job, compute.read_result):
            with self.assertRaises(PublicError) as e:
                fn('bob', one['id'])
            self.assertEqual(e.exception.status, 404)

    def test_legacy_execution_preserves_cpu_and_resolves_gpu_uuid(self):
        self.assertEqual(compute.execution_selection(params={'device': 'cpu', '_device_selection': 'manual'})['mode'], 'cpu')
        self.assertEqual(compute.execution_selection(params={'device': 'cpu', '_device_selection': 'auto'})['mode'], 'auto')
        self.assertEqual(compute.execution_selection(params={'device': '1'})['gpuUuid'], 'GPU-1')
        for selection, params in ((None, {'device': '9'}), ({'mode': 'gpu', 'gpuUuid': None, 'needsSelection': True}, None)):
            with self.assertRaises(PublicError) as exc:
                compute.execution_selection(selection, params)
            self.assertEqual(exc.exception.code, 'GPU_SELECTION_REQUIRED')

    def test_simultaneous_coordinators_never_reserve_one_gpu_twice(self):
        self.create('alice')
        self.create('bob')
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda _: compute.reserve_one(), range(4)))
        rows += [compute.reserve_one()]
        assigned = [r['assigned_gpu_uuid'] for r in rows if r]
        self.assertEqual(len(assigned), 2)
        self.assertEqual(len(set(assigned)), 2)

    def test_token_claim_is_exactly_once_and_cancel_keeps_running_reserved(self):
        job = self.create()
        reserved = compute.reserve_one()
        with compute.registry._connect() as c:
            claimed = compute.claim_job(job['id'], reserved['dispatch_token'], reserved['worker_id'], reserved['assigned_gpu_uuid'], c, memory_cgroup='/system.slice/test-worker')
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed['worker_cgroup'], '/system.slice/test-worker')
        with compute.registry._connect() as c:
            self.assertIsNone(compute.claim_job(job['id'], reserved['dispatch_token'], reserved['worker_id'], reserved['assigned_gpu_uuid'], c))
        self.assertEqual(compute.cancel_job('alice', job['id'])['status'], 'stopping')
        self.assertEqual(compute.get_job('alice', job['id'])['assignedGpu']['uuid'], reserved['assigned_gpu_uuid'])
        compute.finish_job(job['id'], reserved['dispatch_token'], 'completed')
        self.assertEqual(compute.get_job('alice', job['id'])['status'], 'cancelled')

    def test_storage_failure_updates_run_and_releases_reservation(self):
        from unittest.mock import MagicMock
        import compute_worker as worker
        job = self.create(kind='train')
        reserved = compute.reserve_one()
        directory = MagicMock()
        directory.mkdir.side_effect = OSError(28, 'Disk full')
        directory.__truediv__.return_value.write_text.side_effect = OSError(28, 'Disk full')
        with patch.object(worker, 'LEASE', MagicMock()), \
             patch.object(worker, 'WORKER_ID', reserved['worker_id']), \
             patch.object(worker, 'GPU', reserved['assigned_gpu_uuid']), \
             patch.object(worker, 'job_directory', return_value=directory), \
             patch.object(worker, 'gpu_clear', return_value=True), \
             patch.object(worker.threading, 'Thread'), \
             patch.object(worker, 'log'):
            worker.execute(str(job['id']), str(reserved['dispatch_token']))
        self.assertEqual(compute.get_job('alice', job['id'])['status'], 'failed')
        with compute.registry._connect() as c:
            self.assertEqual(c.execute('select status from training_tasks where id=%s', (self.runs['alice'],)).fetchone()['status'], 'failed')
        self.create('bob')
        self.assertEqual(compute.reserve_one()['assigned_gpu_uuid'], reserved['assigned_gpu_uuid'])

    def test_gpu_reservation_survives_stale_job_and_redis_loss(self):
        from compute_coordinator import recover_dispatches
        job = self.create()
        reserved = compute.reserve_one()
        with compute.registry._connect() as c:
            compute.claim_job(job['id'], reserved['dispatch_token'], reserved['worker_id'], reserved['assigned_gpu_uuid'], c)
            c.execute("update compute_jobs set heartbeat_at=now()-interval '40 seconds' where id=%s", (job['id'],))
        recover_dispatches()
        self.assertEqual(compute.get_job('alice', job['id'])['status'], 'recovery_pending')
        self.create('bob')
        self.assertNotEqual(compute.reserve_one()['assigned_gpu_uuid'], reserved['assigned_gpu_uuid'])

    def test_queue_limit_and_active_run_delete_guard(self):
        for _ in range(3): self.create()
        with self.assertRaises(PublicError): self.create()
        with compute.registry._connect() as c:
            with self.assertRaises(Exception):
                c.execute('delete from training_tasks where id=%s', (self.runs['alice'],))

    def test_idempotency_rejects_different_input_or_device(self):
        one = self.create(idempotency_key='input-key')
        with self.assertRaises(PublicError) as exc:
            compute.create_job('alice', self.runs['alice'], 'predict', {'input_sha256':'changed'}, {'mode':'cpu'}, idempotency_key='input-key')
        self.assertEqual(exc.exception.code, 'IDEMPOTENCY_CONFLICT')
        self.assertEqual(self.create(idempotency_key='input-key')['id'], one['id'])

    def test_cleanup_blocks_submissions_before_filesystem_removal(self):
        compute.registry.delete_user_resource_records('alice')
        with self.assertRaises(PublicError) as exc:
            self.create()
        self.assertEqual(exc.exception.code, 'USER_RESOURCES_DELETING')

    def test_terminal_job_does_not_prevent_run_deletion(self):
        job = self.create()
        compute.cancel_job('alice', job['id'])
        with compute.registry._connect() as c:
            c.execute('delete from training_tasks where id=%s', (self.runs['alice'],))
            self.assertIsNone(c.execute('select id from compute_jobs where id=%s', (job['id'],)).fetchone())

    def test_stale_dispatch_token_cannot_claim_reassigned_work(self):
        job = self.create()
        first = compute.reserve_one()
        with compute.registry._connect() as c:
            c.execute("update compute_jobs set status='queued',assigned_gpu_uuid=null,node_id=null,worker_id=null,dispatch_token=null where id=%s", (job['id'],))
        second = compute.reserve_one()
        with compute.registry._connect() as c:
            self.assertIsNone(compute.claim_job(job['id'], first['dispatch_token'], first['worker_id'], first['assigned_gpu_uuid'], c))
            self.assertIsNotNone(compute.claim_job(job['id'], second['dispatch_token'], second['worker_id'], second['assigned_gpu_uuid'], c))

    def test_prediction_files_expire_but_running_files_remain(self):
        from compute_coordinator import cleanup_results
        job = self.create()
        directory = compute.job_directory(job['id'])
        directory.mkdir(parents=True)
        (directory / 'input').write_bytes(b'test')
        compute.cancel_job('alice', job['id'])
        with compute.registry._connect() as c:
            c.execute("update compute_jobs set finished_at=now()-interval '25 hours' where id=%s", (job['id'],))
        cleanup_results()
        self.assertFalse(directory.exists())
        self.assertIsNotNone(compute.get_job('alice', job['id'], public=False)['cleaned_at'])

    def test_version_reservations_never_reuse_cancelled_number(self):
        import label_studio_jobs as jobs
        project = uuid.uuid4()
        with compute.registry._connect() as c:
            c.execute("insert into label_studio_accounts(owner_user_id,ls_user_id) values('alice',1)")
            c.execute("insert into label_studio_projects(id,owner_user_id,ls_project_id,name,kind) values(%s,'alice',1,'Test','object_detection')", (project,))
        with patch.object(jobs, 'enqueue'):
            one = jobs.create('alice', project, 'publish', preview_only=True)
            jobs.cancel('alice', one['id'])
            two = jobs.create('alice', project, 'publish', preview_only=True)
        self.assertGreater(two['version'], one['version'])


if __name__ == '__main__': unittest.main()
