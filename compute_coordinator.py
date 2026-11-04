"""Durable admission and delivery. RQ never owns compute state."""
from __future__ import annotations

import logging
import os
import shutil
import time

import psutil
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from compute_repository import COMPUTE_ROOT, job_directory, queue_name, reserve_one, registry
from settings import REDIS_URL

log = logging.getLogger(__name__)
NODE = os.getenv('COMPUTE_NODE_ID', 'local')


def inventory():
    memory = psutil.virtual_memory()
    with registry._connect() as c:
        c.execute('''insert into compute_nodes(id,ram_total_mb,ram_available_mb,cpu_count,worker_ram_mb,worker_cpus,heartbeat_at)
          values(%s,%s,%s,%s,%s,%s,now()) on conflict(id) do update set ram_total_mb=excluded.ram_total_mb,
          ram_available_mb=excluded.ram_available_mb,cpu_count=excluded.cpu_count,
          worker_ram_mb=excluded.worker_ram_mb,worker_cpus=excluded.worker_cpus,heartbeat_at=now()''',
          (NODE, memory.total // 1048576, memory.available // 1048576, psutil.cpu_count() or 1,
           int(os.getenv('COMPUTE_WORKER_RAM_MB', '8192')), int(os.getenv('COMPUTE_WORKER_CPUS', '4'))))
    try:
        import pynvml as nv
        nv.nvmlInit()
        for index in range(nv.nvmlDeviceGetCount()):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            uuid = nv.nvmlDeviceGetUUID(handle)
            name = nv.nvmlDeviceGetName(handle)
            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            good = True
            try:
                processes = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)}
                processes.update(p.pid for p in nv.nvmlDeviceGetGraphicsRunningProcesses(handle))
                utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
                temperature = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
            except nv.NVMLError:
                good, processes, utilization, temperature = False, set(), None, None
            with registry._connect() as c:
                c.execute('''insert into gpu_devices(uuid,node_id,display_index,name,total_mb,free_mb,
                    utilization,temperature,process_count,telemetry_ok,observed_at)
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()) on conflict(uuid) do update
                    set display_index=excluded.display_index,name=excluded.name,total_mb=excluded.total_mb,
                    free_mb=excluded.free_mb,utilization=excluded.utilization,temperature=excluded.temperature,
                    process_count=excluded.process_count,telemetry_ok=excluded.telemetry_ok,observed_at=now()''',
                    (uuid, NODE, index, name, memory.total // 1048576, memory.free // 1048576,
                     utilization, temperature, len(processes), good))
    except Exception:
        log.exception('GPU inventory unavailable; stale devices cannot receive jobs')
    with registry._connect() as c:
        c.execute('''update training_tasks t set execution=jsonb_build_object('mode','gpu','gpuUuid',g.uuid)
           from gpu_devices g where t.status='draft' and (t.execution->>'needsSelection')::boolean
           and g.node_id=%s and g.display_index::text=t.params->>'device' ''', (NODE,))


def deliver(row, redis):
    # A missing Redis entry can be recreated with the same dispatch token. Claim is a DB CAS.
    try:
        job = Job.fetch(row['rq_job_id'], connection=redis)
        status = job.get_status(refresh=True)
        if str(getattr(status, 'value', status)) not in ('failed', 'finished', 'canceled', 'stopped'):
            return
        job.delete()
    except NoSuchJobError:
        pass
    Queue(queue_name(row['node_id'], row['assigned_gpu_uuid']), connection=redis).enqueue(
        'compute_worker.execute', str(row['id']), str(row['dispatch_token']),
        job_id=row['rq_job_id'], job_timeout=86430 if row['kind'] == 'train' else 630,
        result_ttl=3600, failure_ttl=86400,
    )


def recover_dispatches():
    with registry._connect() as c:
        # Started jobs stay reserved until their supervising worker proves the process stopped.
        c.execute("""update compute_jobs set status='recovery_pending',queue_reason='worker_offline'
              where status in ('running','stopping') and heartbeat_at < now()-interval '30 seconds'""")
        c.execute("""update training_tasks t set status='recovery_pending' from compute_jobs j
              where j.run_id=t.id and j.kind='train' and j.status='recovery_pending'""")
        c.execute("""update compute_jobs j set status='queued',dispatch_token=null,rq_job_id=null,
              node_id=null,assigned_gpu_uuid=null,worker_id=null,queue_reason='worker_offline'
              where j.status='dispatching' and j.dispatched_at < now()-interval '60 seconds'
              and not exists(select 1 from gpu_devices g where g.uuid=j.assigned_gpu_uuid
                    and g.worker_id=j.worker_id and g.worker_heartbeat>now()-interval '30 seconds')
              and not exists(select 1 from compute_nodes n where j.mode='cpu' and n.id=j.node_id
                    and n.cpu_worker_id=j.worker_id and n.cpu_worker_heartbeat>now()-interval '30 seconds')""")


def cleanup_results():
    with registry._connect() as c:
        rows = c.execute("""select id from compute_jobs where
             status in ('completed','failed','cancelled') and cleaned_at is null
             and finished_at < now()-interval '24 hours' for update skip locked""").fetchall()
        for row in rows:
            shutil.rmtree(job_directory(row['id']), ignore_errors=False) if job_directory(row['id']).exists() else None
            c.execute('update compute_jobs set cleaned_at=now(),result_path=null where id=%s', (row['id'],))
        if COMPUTE_ROOT.exists():
            for directory in COMPUTE_ROOT.iterdir():
                if not directory.is_dir() or directory.stat().st_mtime > time.time() - 86400:
                    continue
                try:
                    import uuid
                    job_id = uuid.UUID(directory.name)
                except ValueError:
                    continue
                if not c.execute('select 1 from compute_jobs where id=%s', (job_id,)).fetchone():
                    shutil.rmtree(directory)


def tick(redis):
    inventory()
    recover_dispatches()
    cleanup_results()
    # Keep delivery serialized even if an operator accidentally starts two coordinators.
    with registry._connect() as c:
        if not c.execute('select pg_try_advisory_lock(%s) as ok', (registry._lock_key('compute-delivery'),)).fetchone()['ok']:
            return
        for _ in range(32):
            if not reserve_one():
                break
        pending = c.execute("select * from compute_jobs where status='dispatching' order by dispatched_at").fetchall()
        for row in pending:
            deliver(row, redis)


def main():
    logging.basicConfig(level=logging.INFO)
    redis = Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=3)
    while True:
        try:
            tick(redis)
        except Exception:
            log.exception('Coordinator retrying; DB reservations are preserved')
        time.sleep(5)


if __name__ == '__main__':
    main()
