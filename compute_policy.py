"""Pure admission policy. Locks and state changes belong to compute_repository."""
from datetime import datetime, timezone

ACTIVE = ('dispatching', 'running', 'stopping', 'recovery_pending')
TERMINAL = ('completed', 'failed', 'cancelled')


def recent(timestamp, seconds=30):
    return bool(timestamp and (datetime.now(timezone.utc) - timestamp).total_seconds() < seconds)


def gpu_budget(total_mb):
    return max(0, int(total_mb) - max(1024, int(total_mb * .15)))


def device_state(device, active=False):
    if not device.get('enabled') or device.get('draining'):
        return 'draining'
    if (not recent(device.get('observed_at')) or not recent(device.get('worker_heartbeat'))
            or device.get('health_error') or not device.get('telemetry_ok')):
        return 'offline'
    if device.get('temperature') is not None and device['temperature'] >= 90:
        return 'offline'
    return 'busy' if active or device.get('process_count', 0) else 'ready'


def select_resource(job, devices, nodes, active_jobs, quota):
    gpu_active = [j for j in active_jobs if j.get('assigned_gpu_uuid')]
    if job['mode'] != 'cpu' and sum(j['owner_user_id'] == job['owner_user_id'] for j in gpu_active) >= quota:
        return None, 'user_quota'
    candidates = []
    if job['mode'] == 'cpu':
        for node in nodes:
            if recent(node.get('cpu_worker_heartbeat')) and not node.get('cpu_health_error'):
                if not any(j['node_id'] == node['id'] and j['mode'] == 'cpu' for j in active_jobs):
                    candidates.append((node, None))
    else:
        busy = {j['assigned_gpu_uuid'] for j in gpu_active}
        for gpu in sorted(devices, key=lambda g: (g['total_mb'], g['uuid'])):
            if job['mode'] == 'gpu' and gpu['uuid'] != job['requested_gpu_uuid']:
                continue
            if device_state(gpu, gpu['uuid'] in busy) != 'ready':
                continue
            if job['estimated_vram_mb'] > gpu_budget(gpu['total_mb']):
                continue
            if gpu['free_mb'] < job['estimated_vram_mb'] + max(1024, int(gpu['total_mb'] * .15)):
                continue
            node = next((n for n in nodes if n['id'] == gpu['node_id']), None)
            if node:
                candidates.append((node, gpu))
    if not candidates:
        if job['mode'] == 'cpu':
            return None, 'cpu_busy' if any(recent(n.get('cpu_worker_heartbeat')) and not n.get('cpu_health_error') for n in nodes) else 'worker_offline'
        matching = [g for g in devices if job['mode'] != 'gpu' or g['uuid'] == job['requested_gpu_uuid']]
        states = [device_state(g, g['uuid'] in {j['assigned_gpu_uuid'] for j in gpu_active}) for g in matching]
        if 'busy' in states:
            return None, 'gpu_busy'
        if 'ready' in states:
            return None, 'gpu_memory_busy'
        return None, 'gpu_draining' if states and all(s == 'draining' for s in states) else 'worker_offline'
    for node, gpu in candidates:
        if not recent(node.get('heartbeat_at')):
            continue
        used_ram = sum(j['worker_ram_mb'] for j in active_jobs if j['node_id'] == node['id'])
        # Reserve only the headroom not already reflected in host MemAvailable.
        unallocated_ram = sum(max(0, j['worker_ram_mb'] - max(0, j.get('resident_memory_mb', 0)))
                              for j in active_jobs if j['node_id'] == node['id'])
        used_cpus = sum(j['worker_cpus'] for j in active_jobs if j['node_id'] == node['id'])
        ram, cpus = node['worker_ram_mb'], node['worker_cpus']
        if (used_ram + ram > node['ram_total_mb'] * node['ram_budget_percent'] / 100
                or node['ram_available_mb'] - unallocated_ram < ram + 4096
                or used_cpus + cpus > max(1, node['cpu_count'] - 2)):
            continue
        return {'node_id': node['id'], 'gpu_uuid': gpu['uuid'] if gpu else None,
                'worker_id': gpu['worker_id'] if gpu else node['cpu_worker_id'],
                'ram_mb': ram, 'cpus': cpus}, None
    return None, 'node_resources'


def fair_candidates(jobs, quotas, inference_burst):
    wanted = 'train' if inference_burst >= 3 else 'predict'
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(jobs, key=lambda j: (j['kind'] != wanted,
                  quotas.get(j['owner_user_id'], {}).get('last_dispatched_at') or epoch,
                  j['created_at'], str(j['id'])))
