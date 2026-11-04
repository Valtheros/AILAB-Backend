from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from api_errors import PublicError
from compute_memory import sample_memory
from compute_policy import ACTIVE, TERMINAL, device_state, fair_candidates, gpu_budget, recent, select_resource
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import RUNS_DIR

COMPUTE_ROOT = RUNS_DIR / '.compute'


def enabled():
    return os.getenv('COMPUTE_ENABLED', 'false').lower() == 'true'


def training_record(job_id):
    if not enabled():
        return None
    try:
        job_id = uuid.UUID(str(job_id))
    except ValueError:
        return None
    with registry._connect() as c:
        return c.execute('''select t.*,j.status as compute_status,j.queue_reason,j.assigned_gpu_uuid
              from training_tasks t join compute_jobs j on t.compute_job_id=j.id where j.id=%s''', (job_id,)).fetchone()


def resource_plan(model, params, batch_size, selection):
    from resource_guard import get_resource_profile, validate_resource_plan
    profile = get_resource_profile()
    chosen = execution_selection(selection, params)
    with registry._connect() as c:
        devices = c.execute('select * from gpu_devices' + (' where uuid=%s' if chosen['mode'] == 'gpu' else ''),
                            (chosen['gpuUuid'],) if chosen['mode'] == 'gpu' else ()).fetchall()
    # Auto may wait for a busy large card, but never estimates against unrelated hardware.
    profile['safe_limits']['gpu_vram_mb'] = max((gpu_budget(g['total_mb']) for g in devices), default=0)
    plan = validate_resource_plan(model, {**params, 'device': 'cpu' if chosen['mode'] == 'cpu' else '0'}, batch_size, profile)
    plan['execution'] = chosen
    plan['device'] = 'CPU' if chosen['mode'] == 'cpu' else 'Auto GPU' if chosen['mode'] == 'auto' else chosen['gpuUuid']
    plan['warnings'].append('VRAM is an estimate, not a guarantee against out-of-memory errors.')
    return plan


def execution_selection(selection=None, params=None):
    if selection:
        mode, gpu = selection.get('mode'), selection.get('gpuUuid')
    else:
        params = params or {}
        device = str(params.get('device', 'auto')).lower()
        if params.get('_device_selection') == 'auto' or device == 'auto':
            mode, gpu = 'auto', None
        elif device == 'cpu':
            mode, gpu = 'cpu', None
        else:
            with registry._connect() as c:
                rows = c.execute('select uuid from gpu_devices where display_index::text=%s', (device,)).fetchall()
            if len(rows) != 1:
                raise PublicError('GPU_SELECTION_REQUIRED', 'Select the GPU again. The previous GPU index cannot be identified.', status=409)
            mode, gpu = 'gpu', rows[0]['uuid']
    if mode == 'gpu' and not gpu:
        raise PublicError('GPU_SELECTION_REQUIRED', 'Select the GPU again. The previous GPU index cannot be identified.', status=409)
    if mode not in {'auto', 'gpu', 'cpu'} or (mode == 'gpu' and not gpu) or (mode != 'gpu' and gpu):
        raise PublicError('INVALID_EXECUTION', 'Select Auto GPU, a GPU, or CPU.')
    return {'mode': mode, 'gpuUuid': gpu}


def require_enabled():
    if not enabled():
        raise PublicError('COMPUTE_DISABLED', 'Compute workers are not enabled yet.', status=503)


def job_directory(job_id):
    return contained_path(COMPUTE_ROOT, str(uuid.UUID(str(job_id))))


def list_devices(admin=False):
    require_enabled()
    with registry._connect() as c:
        nodes = c.execute('select * from compute_nodes order by id').fetchall()
        devices = c.execute('select * from gpu_devices order by node_id,display_index').fetchall()
        busy = c.execute('select assigned_gpu_uuid from compute_jobs where status=any(%s)', (list(ACTIVE),)).fetchall()
        quotas = c.execute('''select u.id as owner_user_id,u.email,coalesce(q.max_gpu_jobs,1) as max_gpu_jobs
                 from "user" u left join compute_user_quotas q on q.owner_user_id=u.id order by u.email limit 500''').fetchall() if admin else []
    active = {r['assigned_gpu_uuid'] for r in busy}
    return {'devices': [{'uuid': d['uuid'], 'index': d['display_index'], 'name': d['name'],
             'totalMb': d['total_mb'], 'freeMb': d['free_mb'], 'safeMb': gpu_budget(d['total_mb']),
             'utilization': d['utilization'], 'temperature': d['temperature'],
             'status': device_state(d, d['uuid'] in active), 'error': d['health_error'],
             'nodeId': d['node_id'], 'observedAt': d['observed_at'], 'draining': d['draining'],
             'enabled': d['enabled']} for d in devices],
            'cpuAvailable': any(recent(n['cpu_worker_heartbeat']) and not n['cpu_health_error'] for n in nodes),
            **({'nodes': nodes, 'quotas': quotas} if admin else {})}


def validate_target(selection, estimated_vram_mb, c):
    if selection['mode'] == 'cpu':
        return
    rows = c.execute('select * from gpu_devices' + (' where uuid=%s' if selection['mode'] == 'gpu' else ''),
                     (selection['gpuUuid'],) if selection['mode'] == 'gpu' else ()).fetchall()
    if not rows:
        raise PublicError('GPU_NOT_CONFIGURED', 'No matching GPU is configured. Select CPU or contact an administrator.')
    if not any(gpu_budget(r['total_mb']) >= estimated_vram_mb for r in rows):
        raise PublicError('GPU_MEMORY_LIMIT', 'The selected GPUs cannot fit this configuration. Reduce batch or image size.')


def create_job(owner_id, run_id, kind, payload, selection, *, estimated_vram_mb=0,
               idempotency_key=None, job_id=None, connection=None):
    require_enabled()
    selection = execution_selection(selection)
    own = connection is None
    c = connection or registry._connect()
    try:
        c.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + owner_id),))
        quota = c.execute('select accepting_jobs from compute_user_quotas where owner_user_id=%s', (owner_id,)).fetchone()
        if quota and not quota['accepting_jobs']:
            raise PublicError('USER_RESOURCES_DELETING', 'This account is undergoing resource cleanup. Contact an administrator.', status=409)
        if idempotency_key:
            previous = c.execute('select * from compute_jobs where owner_user_id=%s and idempotency_key=%s',
                                 (owner_id, idempotency_key)).fetchone()
            if previous:
                if (str(previous['run_id']) != str(run_id) or previous['kind'] != kind
                        or previous['payload'] != payload or previous['mode'] != selection['mode']
                        or previous['requested_gpu_uuid'] != selection['gpuUuid']):
                    raise PublicError('IDEMPOTENCY_CONFLICT', 'This request key belongs to a different job.', status=409)
                return dict(previous)
        run = c.execute('select id,status from training_tasks where id=%s and owner_user_id=%s for update', (run_id, owner_id)).fetchone()
        if not run:
            raise PublicError('NOT_FOUND', 'Run not found.', status=404)
        if kind == 'predict' and run['status'] != 'completed':
            raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be tested.', status=409)
        count = c.execute("select count(*) as n from compute_jobs where owner_user_id=%s and kind=%s and status=any(%s)",
                          (owner_id, kind, ['queued', *ACTIVE])).fetchone()['n']
        if count >= (10 if kind == 'train' else 3):
            raise PublicError('QUEUE_LIMIT', 'You have too many pending jobs. Cancel one or wait for it to finish.', status=429)
        validate_target(selection, estimated_vram_mb, c)
        c.execute('insert into compute_user_quotas(owner_user_id) values(%s) on conflict do nothing', (owner_id,))
        row = c.execute('''insert into compute_jobs(id,owner_user_id,run_id,kind,mode,requested_gpu_uuid,
                 payload,estimated_vram_mb,idempotency_key) values(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s) returning *''',
                (job_id or uuid.uuid4(), owner_id, run_id, kind, selection['mode'], selection['gpuUuid'],
                 json.dumps(payload), estimated_vram_mb, idempotency_key)).fetchone()
        if kind == 'train':
            c.execute('update training_tasks set compute_job_id=%s,execution=%s::jsonb where id=%s',
                      (row['id'], json.dumps(selection), run_id))
        if own:
            c.commit()
        return dict(row)
    finally:
        if own:
            c.close()


def public_job(row):
    gpu = row.get('gpu_name')
    return {'id': str(row['id']), 'kind': row['kind'], 'status': row['status'],
            'requestedExecution': {'mode': row['mode'], 'gpuUuid': row['requested_gpu_uuid']},
            'assignedGpu': {'uuid': row['assigned_gpu_uuid'], 'name': gpu} if row['assigned_gpu_uuid'] else None,
            'queueReason': row['queue_reason'], 'errorCode': row['error_code'], 'error': row['error_message'],
            'inputName': row['payload'].get('input_name'), 'inputBytes': row['payload'].get('input_bytes'),
            'createdAt': row['created_at'], 'startedAt': row['started_at'], 'finishedAt': row['finished_at']}


def get_job(owner_id, job_id, *, public=True):
    with registry._connect() as c:
        row = c.execute('''select j.*,g.name as gpu_name from compute_jobs j left join gpu_devices g
                       on g.uuid=j.assigned_gpu_uuid where j.id=%s and j.owner_user_id=%s''', (job_id, owner_id)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Job not found.', status=404)
    return public_job(row) if public else dict(row)


def assert_run_idle(run_id, c):
    if enabled() and c.execute('select 1 from compute_jobs where run_id=%s and status=any(%s) limit 1',
                              (run_id, ['queued', *ACTIVE])).fetchone():
        raise PublicError('RUN_IN_USE', 'Cancel active compute jobs before deleting this run.', status=409)


@contextmanager
def run_guard(owner, run_id):
    with registry._connect() as c:
        c.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + owner),))
        row = c.execute('select id from training_tasks where owner_user_id=%s and id=%s for update', (owner, run_id)).fetchone()
        if not row:
            raise PublicError('NOT_FOUND', 'Run not found.', status=404)
        assert_run_idle(run_id, c)
        yield c


def cancel_job(owner_id, job_id):
    with registry._connect() as c:
        row = c.execute('select * from compute_jobs where id=%s and owner_user_id=%s for update', (job_id, owner_id)).fetchone()
        if not row:
            raise PublicError('NOT_FOUND', 'Job not found.', status=404)
        if row['status'] in ('queued', 'dispatching'):
            c.execute("update compute_jobs set status='cancelled',cancel_requested=true,finished_at=now() where id=%s", (job_id,))
            if row['kind'] == 'train':
                c.execute("update training_tasks set status='cancelled',finished_at=now() where id=%s", (row['run_id'],))
        elif row['status'] in ACTIVE:
            c.execute("update compute_jobs set status='stopping',cancel_requested=true where id=%s", (job_id,))
            if row['kind'] == 'train':
                c.execute("update training_tasks set status='stopping' where id=%s", (row['run_id'],))
    return get_job(owner_id, job_id)


def reserve_one():
    with registry._connect() as c:
        if not c.execute('select pg_try_advisory_xact_lock(%s) as ok', (registry._lock_key('compute-dispatch'),)).fetchone()['ok']:
            return None
        nodes = c.execute('select * from compute_nodes order by id for update').fetchall()
        devices = c.execute('select * from gpu_devices order by uuid for update').fetchall()
        active = c.execute('select * from compute_jobs where status=any(%s)', (list(ACTIVE),)).fetchall()
        sample_memory(nodes, active, os.getenv('COMPUTE_NODE_ID', 'local'))
        pending = c.execute("select * from compute_jobs where status='queued' order by created_at for update skip locked").fetchall()
        quotas = {r['owner_user_id']: r for r in c.execute('select * from compute_user_quotas for update').fetchall()}
        burst = max((n['inference_burst'] for n in nodes), default=0)
        for job in fair_candidates(pending, quotas, burst):
            resource, reason = select_resource(job, devices, nodes, active, quotas.get(job['owner_user_id'], {}).get('max_gpu_jobs', 1))
            if not resource:
                c.execute('update compute_jobs set queue_reason=%s where id=%s', (reason, job['id']))
                continue
            token = uuid.uuid4()
            rq_id = f'compute-{job["id"]}-{token}'
            row = c.execute('''update compute_jobs set status='dispatching',node_id=%s,assigned_gpu_uuid=%s,
                worker_id=%s,worker_ram_mb=%s,worker_cpus=%s,dispatch_token=%s,rq_job_id=%s,
                dispatched_at=now(),queue_reason=null where id=%s returning *''',
                (resource['node_id'], resource['gpu_uuid'], resource['worker_id'], resource['ram_mb'],
                 resource['cpus'], token, rq_id, job['id'])).fetchone()
            c.execute('update compute_user_quotas set last_dispatched_at=now() where owner_user_id=%s', (job['owner_user_id'],))
            c.execute('update compute_nodes set inference_burst=%s where id=%s',
                      (min(burst + 1, 3) if job['kind'] == 'predict' else 0, resource['node_id']))
            return dict(row)


def queue_name(node_id, gpu_uuid):
    return 'compute_' + registry._lock_key(f'{node_id}:{gpu_uuid or "cpu"}').__str__().replace('-', 'n')


def claim_job(job_id, token, worker_id, gpu_uuid, c, *, memory_cgroup=None):
    row = c.execute('''update compute_jobs set status='running',started_at=now(),heartbeat_at=now(),worker_cgroup=%s
          where id=%s and dispatch_token=%s and worker_id=%s and assigned_gpu_uuid is not distinct from %s
          and status='dispatching' and not cancel_requested returning *''', (memory_cgroup, job_id, token, worker_id, gpu_uuid)).fetchone()
    if row and row['kind'] == 'train':
        c.execute("update training_tasks set status='running',updated_at=now() where id=%s", (row['run_id'],))
    return dict(row) if row else None


def finish_job(job_id, token, status, *, error_code=None, error_message=None, result_path=None):
    with registry._connect() as c:
        row = c.execute('''select * from compute_jobs where id=%s and dispatch_token=%s and status=any(%s) for update''',
                         (job_id, token, list(ACTIVE))).fetchone()
        if not row:
            return
        if row['cancel_requested']:
            status = 'cancelled'
        c.execute('''update compute_jobs set status=%s,error_code=%s,error_message=%s,result_path=%s,
                     finished_at=now(),heartbeat_at=now() where id=%s''', (status, error_code, error_message, result_path, job_id))
        if row['kind'] == 'train':
            projection = 'stopped' if status == 'cancelled' and row['started_at'] else status
            c.execute('update training_tasks set status=%s,error_detail=%s,finished_at=now(),updated_at=now() where id=%s',
                      (projection, error_message, row['run_id']))


def read_result(owner_id, job_id):
    row = get_job(owner_id, job_id, public=False)
    if row['status'] != 'completed' or row['kind'] != 'predict':
        raise PublicError('RESULT_NOT_READY', 'Prediction is not ready.', status=409)
    path = job_directory(job_id) / 'result.json'
    if not path.is_file():
        raise PublicError('RESULT_EXPIRED', 'This result expired. Upload the image again.', status=410)
    return json.loads(path.read_text())
