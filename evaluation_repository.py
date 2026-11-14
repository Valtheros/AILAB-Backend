"""Durable evaluation requests; compute jobs provide resource scheduling."""
from __future__ import annotations

import json
import logging
import uuid

from api_errors import PublicError
import compute_repository as compute
from resource_guard import estimate_training_memory
from resource_repository import resource_repository as registry

log = logging.getLogger(__name__)


def public_evaluation(row):
    return {'id': str(row['id']), 'runId': str(row['run_id']), 'datasetVersionId': str(row['dataset_version_id']),
            'jobId': str(row['compute_job_id']) if row.get('compute_job_id') else None,
            'origin': row['origin'], 'status': row['status'], 'execution': row['execution'],
            'datasetContentSha256': row.get('dataset_content_sha256'), 'checkpointSha256': row.get('checkpoint_sha256'),
            'error': row.get('error_message'), 'createdAt': row['created_at'], 'finishedAt': row.get('finished_at')}


def create_manual(owner_id, run_id, dataset_id, selection, key):
    selection = compute.execution_selection(selection)
    if not key or len(key) > 128:
        raise PublicError('IDEMPOTENCY_REQUIRED', 'A valid request key is required.')
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + owner_id),))
        previous = connection.execute('select * from evaluations where owner_user_id=%s and idempotency_key=%s',
                                      (owner_id, key)).fetchone()
        if previous:
            if (str(previous['run_id']) != str(run_id) or str(previous['dataset_version_id']) != str(dataset_id)
                    or previous['execution'] != selection):
                raise PublicError('IDEMPOTENCY_CONFLICT', 'This request key belongs to a different evaluation.', status=409)
            return public_evaluation(previous)
        run = connection.execute('select id,status from training_tasks where id=%s and owner_user_id=%s for update',
                                 (run_id, owner_id)).fetchone()
        dataset = connection.execute("select id,content_sha256 from datasets where id=%s and owner_user_id=%s and status='active'",
                                     (dataset_id, owner_id)).fetchone()
        if not run or not dataset:
            raise PublicError('NOT_FOUND', 'Run or dataset version not found.', status=404)
        if run['status'] != 'completed':
            raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be evaluated.', status=409)
        pending = connection.execute('''select
            (select count(*) from compute_jobs where owner_user_id=%s and kind in ('train','evaluate') and status=any(%s)) +
            (select count(*) from evaluations where owner_user_id=%s and status='awaiting_queue') as count''',
            (owner_id, ['queued', *compute.ACTIVE], owner_id)).fetchone()['count']
        if pending >= 10:
            raise PublicError('QUEUE_LIMIT', 'You have too many pending jobs. Cancel one or wait for it to finish.', status=429)
        row = connection.execute('''insert into evaluations(owner_user_id,run_id,dataset_version_id,origin,execution,
            dataset_content_sha256,idempotency_key) values(%s,%s,%s,'manual',%s::jsonb,%s,%s) returning *''',
            (owner_id, run_id, dataset_id, json.dumps(selection), dataset['content_sha256'], key)).fetchone()
        return public_evaluation(row)


def enqueue_pending(limit=32):
    if not compute.enabled():
        return
    with registry._connect() as connection:
        candidates = connection.execute("select id,owner_user_id from evaluations where status='awaiting_queue' order by created_at limit %s", (limit,)).fetchall()
    for candidate in candidates:
        try:
            with registry._connect() as connection:
                owner_id = candidate['owner_user_id']
                connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + owner_id),))
                row = connection.execute('''select e.*,t.run_slug,t.task_type,t.model_type,t.params,d.status as dataset_status
                    from evaluations e join training_tasks t on t.id=e.run_id join datasets d on d.id=e.dataset_version_id
                    where e.id=%s and e.owner_user_id=%s for update of e''', (candidate['id'], owner_id)).fetchone()
                if not row or row['status'] != 'awaiting_queue':
                    continue
                if row['dataset_status'] != 'active':
                    connection.execute("update evaluations set status='failed',error_message='Dataset version is no longer available.',finished_at=now(),updated_at=now() where id=%s", (row['id'],))
                    from notification_service import emit
                    emit(connection, owner_id, f"evaluation:{row['id']}:failed", 'evaluate', 'failed', f"/tasks/{row['run_id']}")
                    continue
                try:
                    job = compute.create_job(owner_id, row['run_id'], 'evaluate',
                        {'evaluation_id': str(row['id']), 'project_name': row['run_slug'],
                         'task_type': row['task_type'], 'model_type': row['model_type']}, row['execution'],
                        idempotency_key=f"evaluation:{row['id']}", dataset_version_id=row['dataset_version_id'],
                        estimated_vram_mb=estimate_training_memory(row['model_type'], row['params'] or {}, 1),
                        job_id=uuid.uuid4(), connection=connection)
                except PublicError as exc:
                    if exc.code == 'QUEUE_LIMIT':
                        continue
                    connection.execute("update evaluations set status='failed',error_message=%s,finished_at=now(),updated_at=now() where id=%s", (str(exc), row['id']))
                    from notification_service import emit
                    emit(connection, owner_id, f"evaluation:{row['id']}:failed", 'evaluate', 'failed', f"/tasks/{row['run_id']}")
                    continue
                connection.execute("update evaluations set status='queued',compute_job_id=%s,updated_at=now() where id=%s", (job['id'], row['id']))
        except Exception:
            log.exception('Evaluation enqueue will retry: %s', candidate['id'])


def cancel(owner_id, evaluation_id):
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + owner_id),))
        row = connection.execute('select * from evaluations where id=%s and owner_user_id=%s for update',
                                 (evaluation_id, owner_id)).fetchone()
        if not row:
            raise PublicError('NOT_FOUND', 'Evaluation not found.', status=404)
        if row['status'] == 'awaiting_queue':
            row = connection.execute("update evaluations set status='cancelled',finished_at=now(),updated_at=now() where id=%s returning *", (evaluation_id,)).fetchone()
        job_id = row['compute_job_id']
    if job_id and row['status'] in {'queued', 'running', 'stopping'}:
        compute.cancel_job(owner_id, job_id)
        with registry._connect() as connection:
            row = connection.execute('select * from evaluations where id=%s and owner_user_id=%s', (evaluation_id, owner_id)).fetchone()
    return public_evaluation(row)
