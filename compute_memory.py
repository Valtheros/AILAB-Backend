"""Read-only cgroup accounting for local compute admission."""
from pathlib import Path

import psutil

CGROUP_ROOT = Path('/sys/fs/cgroup')


def worker_cgroup():
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                path = line[3:]
                return path if path != '/' else None
    except OSError:
        pass
    return None


def anonymous_memory_mb(cgroup):
    if not cgroup or not isinstance(cgroup, str) or not cgroup.startswith('/'):
        return 0
    try:
        root = CGROUP_ROOT.resolve()
        directory = (root / cgroup.lstrip('/')).resolve()
        if directory == root or not directory.is_relative_to(root):
            return 0
        # Never credit the host/root cgroup or an unbounded worker.
        if int((directory / 'memory.max').read_text()) <= 0:
            return 0
        stats = dict(line.split() for line in (directory / 'memory.stat').read_text().splitlines())
        # File cache is reclaimable and already counted in MemAvailable.
        return max(0, int(stats['anon'])) // 1048576
    except (OSError, ValueError, KeyError):
        return 0


def sample_memory(nodes, active_jobs, node_id):
    node = next((n for n in nodes if n['id'] == node_id), None)
    if node is None:
        return
    before = psutil.virtual_memory().available // 1048576
    seen = set()
    for job in active_jobs:
        if job['node_id'] != node_id:
            continue
        cgroup = job.get('worker_cgroup')
        job['resident_memory_mb'] = 0
        if cgroup and cgroup not in seen and job['status'] != 'dispatching':
            job['resident_memory_mb'] = anonymous_memory_mb(cgroup)
            seen.add(cgroup)
    after = psutil.virtual_memory().available // 1048576
    node['ram_available_mb'] = min(node['ram_available_mb'], before, after)
