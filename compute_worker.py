"""RQ transport with a supervised process and a PostgreSQL device lease."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
import threading

from redis import Redis
from rq import SimpleWorker

from compute_memory import worker_cgroup
from compute_repository import claim_job, finish_job, job_directory, queue_name, registry
from settings import REDIS_URL

log = logging.getLogger(__name__)
NODE = os.getenv('COMPUTE_NODE_ID', 'local')
GPU = os.getenv('COMPUTE_GPU_UUID') or None
WORKER_ID = str(uuid.uuid4())
LEASE = None
CHILD = None


def gpu_clear():
    if not GPU:
        return True
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(GPU)
    return not nv.nvmlDeviceGetComputeRunningProcesses(handle) and not nv.nvmlDeviceGetGraphicsRunningProcesses(handle)


def heartbeat(job_id=None, token=None, health_error=None):
    # Checking the lease connection prevents a reconnected worker from assuming it still owns a GPU.
    LEASE.execute('select 1')
    with registry._connect() as c:
        if GPU:
            c.execute('''update gpu_devices set worker_id=%s,worker_heartbeat=now(),health_error=%s where uuid=%s''',
                      (WORKER_ID, health_error, GPU))
        else:
            c.execute('''update compute_nodes set cpu_worker_id=%s,cpu_worker_heartbeat=now(),cpu_health_error=%s where id=%s''',
                      (WORKER_ID, health_error, NODE))
        if job_id:
            return c.execute('''update compute_jobs set heartbeat_at=now() where id=%s and dispatch_token=%s
                   returning cancel_requested,status''', (job_id, token)).fetchone()


def terminate_child():
    global CHILD
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        try:
            CHILD.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=5)
    # Trainers may create subprocesses that outlive their parent.
    if CHILD:
        try:
            os.killpg(CHILD.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def execute(job_id, token):
    global CHILD
    if LEASE is None:
        raise RuntimeError('Compute jobs must run in the leased compute worker.')
    row = None
    directory = None
    outcome = {'status': 'failed', 'error_code': 'WORKER_INTERRUPTED', 'error_message': 'The worker was interrupted. Train again to create a new run.'}
    last_db_success = time.monotonic()
    db_lost = False
    watchdog_stop = threading.Event()

    def database_watchdog():
        # statement_timeout cannot protect against a network black hole or a paused database.
        while not watchdog_stop.wait(.5):
            if time.monotonic() - last_db_success >= 10:
                try:
                    if CHILD and CHILD.poll() is None:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                finally:
                    os._exit(1)

    watchdog = threading.Thread(target=database_watchdog, daemon=True)
    watchdog.start()
    try:
        try:
            with registry._connect() as c:
                row = claim_job(job_id, token, WORKER_ID, GPU, c, memory_cgroup=worker_cgroup())
        except Exception:
            # The claim may have committed even if its acknowledgement was lost.
            row = None
            log.exception('Uncertain compute claim; restarting for lease recovery')
            os._exit(1)
        if not row:
            return {'duplicate': True}
        last_db_success = time.monotonic()
        directory = job_directory(job_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.exception('Could not create compute job directory')
            outcome = {'status': 'failed', 'error_code': 'WORKER_STORAGE_ERROR',
                       'error_message': 'The worker could not prepare job storage. Check free disk space and directory permissions, then retry.'}
            return outcome
        if GPU:
            import pynvml as nv
            nv.nvmlInit()
            memory = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByUUID(GPU))
            required = (row['estimated_vram_mb'] + max(1024, int(memory.total / 1048576 * .15))) * 1048576
            if not gpu_clear() or memory.free < required:
                outcome = {'status': 'failed', 'error_code': 'GPU_UNAVAILABLE', 'error_message': 'The GPU changed after reservation. Retry after checking its load.'}
                return outcome
        env = {**os.environ, 'COMPUTE_EXEC_DEVICE': 'cuda:0' if GPU else 'cpu',
               'OMP_NUM_THREADS': str(row['worker_cpus']), 'MKL_NUM_THREADS': str(row['worker_cpus'])}
        CHILD = subprocess.Popen([sys.executable, '-m', 'compute_executor', job_id, token], env=env, cwd=directory, start_new_session=True)
        deadline = time.monotonic() + (86400 if row['kind'] == 'train' else 600)
        while CHILD.poll() is None:
            try:
                current = heartbeat(job_id, token)
                last_db_success = time.monotonic()
                if not current or current['cancel_requested']:
                    terminate_child()
                    outcome = {'status': 'cancelled'}
                    break
            except Exception:
                if time.monotonic() - last_db_success >= 10:
                    db_lost = True
                    terminate_child()
                    break
            if time.monotonic() > deadline:
                terminate_child()
                outcome = {'status': 'failed', 'error_code': 'COMPUTE_TIMEOUT', 'error_message': 'The job exceeded its time limit.'}
                break
            # Short polls keep cancellation responsive without exceeding the five-second heartbeat interval.
            time.sleep(1)
        forced_stop = outcome['status'] == 'cancelled' or outcome.get('error_code') == 'COMPUTE_TIMEOUT'
        if (directory / f'{token}.json').is_file() and not db_lost and not forced_stop:
            outcome = json.loads((directory / f'{token}.json').read_text())
        elif CHILD.returncode == -9 and not forced_stop:
            outcome = {'status': 'failed', 'error_code': 'WORKER_KILLED', 'error_message': 'The process was killed, possibly by the RAM limit. Reduce the configuration and train again.'}
        return outcome
    except Exception:
        log.exception('Compute supervisor failed')
        return outcome
    finally:
        try:
            if row:
                terminate_child()
                CHILD = None
                if db_lost:
                    os._exit(1)
                clear = False
                for _ in range(10):
                    if gpu_clear():
                        clear = True
                        break
                    time.sleep(1)
                if not clear:
                    raise RuntimeError('GPU processes remain after job exit. Device quarantined.')
                if directory is not None:
                    try:
                        (directory / f'released-{token}.json').write_text(json.dumps(outcome))
                    except OSError:
                        # A full disk must not prevent the authoritative DB status update.
                        log.warning('Could not write release proof for %s', job_id, exc_info=True)
                finish_job(job_id, token, **outcome)
        except Exception:
            log.exception('Could not confirm compute release; restarting for lease recovery')
            os._exit(1)
        finally:
            watchdog_stop.set()
            watchdog.join(timeout=1)


def preflight():
    code = '''import os, torch, torchvision
gpu = bool(os.getenv('COMPUTE_GPU_UUID'))
if gpu and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
    raise RuntimeError('Worker must see exactly its assigned CUDA GPU')
device='cuda:0' if gpu else 'cpu'
x=torch.ones(4,device=device); assert x.sum().item()==4
torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device=device),torch.tensor([1.],device=device),.5)
if gpu: torch.cuda.synchronize()
print(torch.__version__,torchvision.__version__,device)
'''
    probe = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=90)
    if probe.returncode:
        raise RuntimeError(probe.stderr[-2000:])
    log.info('Preflight: %s', probe.stdout.strip())


def recover_stopped_jobs():
    # The replacement holds the exclusive lease and waits beyond the old supervisor's DB-loss kill deadline.
    time.sleep(30)
    if not gpu_clear():
        raise RuntimeError('GPU has external or unreaped processes; recovery requires operator inspection')
    with registry._connect() as c:
        rows = c.execute('''select * from compute_jobs where node_id=%s
          and assigned_gpu_uuid is not distinct from %s and status in ('running','stopping','recovery_pending')
          and worker_id<>%s''', (NODE, GPU, WORKER_ID)).fetchall()
    for row in rows:
        finish_job(row['id'], row['dispatch_token'], 'failed', error_code='WORKER_INTERRUPTED',
                   error_message='The worker restarted. Saved checkpoints and logs are preserved. Train again to create a new run.')


def main():
    global LEASE
    logging.basicConfig(level=logging.INFO)
    # Keep a session-level lease open, not a long-running transaction.
    LEASE = registry._connect()
    LEASE.autocommit = True
    LEASE.execute('set statement_timeout=3000')
    if not LEASE.execute('select pg_try_advisory_lock(%s) as ok', (registry._lock_key(f'compute-worker:{NODE}:{GPU or "cpu"}'),)).fetchone()['ok']:
        raise RuntimeError('Another worker owns this device')
    try:
        preflight()
        recover_stopped_jobs()
    except Exception as exc:
        reason = str(exc)
        heartbeat(health_error=reason)
        log.exception('Worker failed preflight; not accepting jobs')
        while True:
            heartbeat(health_error=reason)
            time.sleep(5)
    redis = Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=3)
    while True:
        heartbeat()
        with registry._connect() as c:
            unfinished = c.execute("select id,dispatch_token from compute_jobs where worker_id=%s and status in ('running','stopping','recovery_pending')", (WORKER_ID,)).fetchall()
        for row in unfinished:
            proof = job_directory(row['id']) / f'released-{row["dispatch_token"]}.json'
            if proof.is_file() and gpu_clear():
                finish_job(row['id'], row['dispatch_token'], **json.loads(proof.read_text()))
        try:
            SimpleWorker([queue_name(NODE, GPU)], connection=redis, name=WORKER_ID,
                         job_monitoring_interval=5).work(burst=True, max_jobs=1, logging_level='WARNING')
        except Exception:
            log.exception('RQ unavailable; PostgreSQL retains queued work')
        time.sleep(2)


if __name__ == '__main__':
    # RQ imports the callable by module name; keep the lease in that same module.
    sys.modules['compute_worker'] = sys.modules[__name__]
    main()
