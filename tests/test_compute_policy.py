import unittest
from datetime import datetime, timedelta, timezone
from compute_policy import device_state, fair_candidates, gpu_budget, select_resource


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.node = dict(id='local', ram_total_mb=24576, ram_available_mb=21000, ram_budget_percent=70,
                         worker_ram_mb=8192, worker_cpus=4, cpu_count=24, heartbeat_at=self.now,
                         cpu_worker_id='cpu', cpu_worker_heartbeat=self.now, cpu_health_error=None)
        self.devices = [dict(uuid=f'GPU-{i}', name=f'Test {i}', total_mb=ram, free_mb=ram, enabled=True,
                            draining=False, health_error=None, telemetry_ok=True, process_count=0,
                            worker_id=f'worker-{i}', worker_heartbeat=self.now, observed_at=self.now, node_id='local')
                        for i, ram in enumerate((8192, 12288, 24576, 49152))]
        self.job = dict(id='job', owner_user_id='alice', mode='auto', requested_gpu_uuid=None,
                        estimated_vram_mb=5000, kind='train', created_at=self.now)

    def select(self, **kwargs):
        return select_resource(kwargs.get('job', self.job), self.devices, [self.node], kwargs.get('active', []), kwargs.get('quota', 1))

    def test_smallest_eligible_gpu_not_lowest_index(self):
        self.devices.reverse()
        self.assertEqual(self.select()[0]['gpu_uuid'], 'GPU-0')
        self.job['estimated_vram_mb'] = 10000
        self.assertEqual(self.select()[0]['gpu_uuid'], 'GPU-1')

    def test_pinned_gpu_never_falls_back(self):
        self.job.update(mode='gpu', requested_gpu_uuid='GPU-0')
        self.devices[0]['process_count'] = 1
        self.assertIsNone(self.select()[0])
        self.assertEqual(self.select()[1], 'gpu_busy')

    def test_index_changes_do_not_change_uuid_selection(self):
        self.job.update(mode='gpu', requested_gpu_uuid='GPU-1')
        for i, gpu in enumerate(self.devices): gpu['display_index'] = 3 - i
        self.devices.reverse()
        self.assertEqual(self.select()[0]['gpu_uuid'], 'GPU-1')

    def test_ram_reservations_are_subtracted_from_available_memory(self):
        self.node['ram_available_mb'] = 17000
        active = [dict(self.job, owner_user_id='bob', assigned_gpu_uuid='GPU-0', node_id='local', worker_ram_mb=8192, worker_cpus=4)]
        self.assertEqual(self.select(active=active)[1], 'node_resources')

    def test_resident_ram_is_not_reserved_twice(self):
        self.node['ram_available_mb'] = 13312
        active = [dict(self.job, owner_user_id='bob', assigned_gpu_uuid='GPU-0', node_id='local',
                       worker_ram_mb=8192, worker_cpus=4, resident_memory_mb=8192)]
        self.assertEqual(self.select(active=active)[0]['gpu_uuid'], 'GPU-1')
        self.job['mode'] = 'cpu'
        self.assertIsNotNone(self.select(active=active)[0])
        self.node['ram_available_mb'] = 12000
        self.assertEqual(self.select(active=active)[1], 'node_resources')

    def test_unused_ram_headroom_is_still_reserved(self):
        self.node['ram_available_mb'] = 17000
        active = [dict(self.job, owner_user_id='bob', assigned_gpu_uuid='GPU-0', node_id='local',
                       worker_ram_mb=8192, worker_cpus=4, resident_memory_mb=2048)]
        self.assertEqual(self.select(active=active)[1], 'node_resources')

    def test_resident_memory_does_not_raise_total_budget(self):
        active = [dict(self.job, owner_user_id='bob', assigned_gpu_uuid=f'GPU-{i}', node_id='local',
                       worker_ram_mb=8192, worker_cpus=4, resident_memory_mb=8192) for i in (0, 1)]
        self.assertEqual(self.select(active=active)[1], 'node_resources')

    def test_external_process_and_stale_heartbeat_block_admission(self):
        for gpu in self.devices:
            gpu['process_count'] = 1
        self.assertIsNone(self.select()[0])
        for gpu in self.devices:
            gpu.update(process_count=0, worker_heartbeat=self.now - timedelta(seconds=31))
        self.assertIsNone(self.select()[0])

    def test_drain_preserves_busy_reservation(self):
        self.devices[0]['draining'] = True
        self.assertEqual(device_state(self.devices[0], True), 'draining')
        self.assertEqual(self.select()[0]['gpu_uuid'], 'GPU-1')

    def test_quota_includes_prediction_and_training(self):
        active = [dict(self.job, assigned_gpu_uuid='GPU-0', node_id='local', worker_ram_mb=8192, worker_cpus=4, kind='predict')]
        self.assertEqual(self.select(active=active)[1], 'user_quota')
        self.assertEqual(self.select(active=active, quota=2)[0]['gpu_uuid'], 'GPU-1')

    def test_two_workers_fit_but_three_do_not_on_24gb(self):
        active = [dict(self.job, owner_user_id='bob', assigned_gpu_uuid='GPU-0', node_id='local', worker_ram_mb=8192, worker_cpus=4)]
        self.assertIsNotNone(self.select(active=active)[0])
        active.append(dict(active[0], assigned_gpu_uuid='GPU-1'))
        self.assertEqual(self.select(active=active)[1], 'node_resources')

    def test_available_ram_floor_and_cpu_share_budget(self):
        self.node['ram_available_mb'] = 12000
        self.assertEqual(self.select()[1], 'node_resources')
        self.job['mode'] = 'cpu'
        self.assertEqual(self.select()[1], 'node_resources')

    def test_vram_margin(self):
        self.assertEqual(gpu_budget(4096), 3072)
        self.assertEqual(gpu_budget(12288), 10445)

    def test_user_fairness_and_three_prediction_burst(self):
        bob = dict(self.job, id='bob', owner_user_id='bob', kind='predict')
        alice = dict(self.job, id='alice', kind='predict')
        train = dict(self.job, id='train')
        quota = {'alice': {'last_dispatched_at': self.now}, 'bob': {'last_dispatched_at': None}}
        self.assertEqual(fair_candidates([alice, bob, train], quota, 0)[0]['id'], 'bob')
        self.assertEqual(fair_candidates([alice, bob, train], quota, 3)[0]['id'], 'train')


if __name__ == '__main__':
    unittest.main()
