import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import MagicMock

import compute_memory as memory
import compute_repository as compute


class MemoryTests(unittest.TestCase):
    def test_cpu_is_not_ready_when_coordinator_heartbeat_is_stale(self):
        now = datetime.now(timezone.utc)
        node = {'heartbeat_at': now - timedelta(seconds=31), 'cpu_worker_heartbeat': now, 'cpu_health_error': None}
        context = MagicMock()
        context.__enter__.return_value.execute.return_value.fetchall.side_effect = [[node], [], []]
        with patch.dict('os.environ', {'COMPUTE_ENABLED': 'true'}), patch.object(compute.registry, '_connect', return_value=context):
            result = compute.list_devices()
        self.assertFalse(result['coordinatorAvailable'])
        self.assertFalse(result['cpuAvailable'])

    def test_cpu_resource_plan_does_not_claim_gpu_memory_warning(self):
        context = MagicMock()
        context.__enter__.return_value.execute.return_value.fetchall.return_value = []
        profile = {'safe_limits': {'gpu_vram_mb': 0}}
        plan = {'warnings': []}
        with patch.object(compute.registry, '_connect', return_value=context), \
             patch('resource_guard.get_resource_profile', return_value=profile), \
             patch('resource_guard.validate_resource_plan', return_value=plan):
            result = compute.resource_plan('resnet', {}, 2, {'mode': 'cpu'})
        self.assertEqual(result['device'], 'CPU')
        self.assertEqual(result['warnings'], [])
    def test_only_anonymous_ram_is_credited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / 'worker'
            group.mkdir()
            (group / 'memory.max').write_text(str(8192 * 1048576))
            (group / 'memory.stat').write_text(f'file {4096 * 1048576}\nanon {2048 * 1048576}\n')
            with patch.object(memory, 'CGROUP_ROOT', root):
                self.assertEqual(memory.anonymous_memory_mb('/worker'), 2048)
                for path in ('/', '/missing', None, '/../outside', 'worker'):
                    self.assertEqual(memory.anonymous_memory_mb(path), 0)
                (group / 'memory.max').write_text('max')
                self.assertEqual(memory.anonymous_memory_mb('/worker'), 0)
                (group / 'memory.max').write_text('1000')
                (group / 'memory.stat').write_text('invalid')
                self.assertEqual(memory.anonymous_memory_mb('/worker'), 0)

    def test_worker_identifies_its_host_cgroup(self):
        with patch.object(Path, 'read_text', return_value='0::/system.slice/docker-test.scope\n'):
            self.assertEqual(memory.worker_cgroup(), '/system.slice/docker-test.scope')
        with patch.object(Path, 'read_text', return_value='0::/\n'):
            self.assertIsNone(memory.worker_cgroup())
        with patch.object(Path, 'read_text', side_effect=OSError):
            self.assertIsNone(memory.worker_cgroup())

    def test_snapshot_uses_fresh_memory_and_credits_each_group_once(self):
        nodes = [dict(id='local', ram_available_mb=21000)]
        jobs = [dict(node_id='local', status='running', worker_cgroup='/a'),
                dict(node_id='local', status='running', worker_cgroup='/a'),
                dict(node_id='local', status='dispatching', worker_cgroup='/b'),
                dict(node_id='local', status='recovery_pending', worker_cgroup=None),
                dict(node_id='remote', status='running', worker_cgroup='/c')]
        with patch.object(memory.psutil, 'virtual_memory', side_effect=[SimpleNamespace(available=14000 * 1048576), SimpleNamespace(available=13000 * 1048576)]), \
             patch.object(memory, 'anonymous_memory_mb', return_value=8192) as measure:
            memory.sample_memory(nodes, jobs, 'local')
        self.assertEqual(nodes[0]['ram_available_mb'], 13000)
        self.assertEqual([j.get('resident_memory_mb', 0) for j in jobs], [8192, 0, 0, 0, 0])
        measure.assert_called_once_with('/a')
