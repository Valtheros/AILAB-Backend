"""Durable import/publish jobs. PostgreSQL owns state; RQ only schedules execution."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from redis import Redis
from rq import Queue, Worker
from rq.job import Job

from dataset_storage import owner_dataset_path
from dataset_utils import inspect_dataset, validate_dataset_for_upload, dataset_workflow_metadata, safe_dataset_name
from label_studio_client import BridgeError, bridge
from label_studio_export import SnapshotExporter
from model_catalog import get_catalog
from resource_repository import resource_repository as registry
from security_utils import contained_path, safe_extract_zip
from settings import DATASET_DIR, REDIS_URL
from api_errors import error_info

logger = logging.getLogger(__name__)
QUEUE = 'label_studio'
WORK_DIR = DATASET_DIR / '.label-studio-work'


def operation_error(exc):
    if isinstance(exc, BridgeError):
        if exc.status in {400, 404, 409, 422}:
            return str(exc)[:500]
        return 'Label Studio is unavailable. Your saved work has not been removed. Please try again later.'
    if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError, zipfile.BadZipFile)):
        return str(exc)[:500]
    return 'Operation failed. Retry after checking the service; details are in the server log.'


def operation(owner, operation_id):
    with registry._connect() as c:
        row = c.execute('select * from label_studio_operations where owner_user_id=%s and id=%s', (owner, operation_id)).fetchone()
    if not row:
        raise FileNotFoundError('Operation not found.')
    return public_operation(row)


def public_operation(row):
    result = dict(row)
    result['requestId'] = str(row['id'])
    result.pop('upload_path', None)
    result.pop('rq_job_id', None)
    return result


def enqueue(operation_id):
    try:
        q = Queue(QUEUE, connection=Redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=5))
        job_id = f'ls-{operation_id}-{uuid.uuid4().hex[:8]}'
        with registry._connect() as c:
            row = c.execute("update label_studio_operations set rq_job_id=%s where id=%s and status='queued' and not cancel_requested returning id", (job_id, operation_id)).fetchone()
            if not row:
                return
        q.enqueue('label_studio_jobs.run', str(operation_id), job_id=job_id, job_timeout=86400,
                  result_ttl=3600, failure_ttl=86400)
    except Exception:
        logger.warning('Label Studio operation will wait for queue recovery.', exc_info=True)


def create(owner, project_id, kind, upload_path=None, *, preview_only=False, operation_id=None):
    with registry._connect() as c:
        project = c.execute('select * from label_studio_projects where owner_user_id=%s and id=%s and deleted_at is null for update', (owner, project_id)).fetchone()
        if not project:
            raise FileNotFoundError('Project not found.')
        if operation_id:
            prior = c.execute('select * from label_studio_operations where id=%s', (operation_id,)).fetchone()
            if prior:
                if prior['owner_user_id'] != owner or str(prior['project_id']) != str(project_id):
                    raise FileNotFoundError('Operation not found.')
                if prior['kind'] != kind or prior['preview_only'] != preview_only:
                    raise ValueError('Transfer ID belongs to a different operation.')
                return public_operation(prior)
        if c.execute("select 1 from label_studio_operations where project_id=%s and status in ('queued','running')", (project_id,)).fetchone():
            raise FileExistsError('This project already has an active operation.')
        version = None
        if kind == 'publish':
            version = c.execute('update label_studio_projects set reserved_version=greatest(reserved_version,published_version)+1 where id=%s returning reserved_version', (project_id,)).fetchone()['reserved_version']
        row = c.execute('insert into label_studio_operations(id,owner_user_id,project_id,kind,upload_path,preview_only,version) values(%s,%s,%s,%s,%s,%s,%s) returning id',
                        (operation_id or uuid.uuid4(), owner, project_id, kind, str(upload_path) if upload_path else None, preview_only, version)).fetchone()
    enqueue(row['id'])
    return operation(owner, row['id'])


def retry(owner, operation_id):
    with registry._connect() as c:
        reference = c.execute('select project_id from label_studio_operations where id=%s and owner_user_id=%s', (operation_id, owner)).fetchone()
        if not reference:
            raise FileNotFoundError('Operation not found.')
        project = c.execute('select id from label_studio_projects where id=%s and deleted_at is null for update', (reference['project_id'],)).fetchone()
        if not project:
            raise FileNotFoundError('Project was removed.')
        row = c.execute('select * from label_studio_operations where id=%s and owner_user_id=%s for update', (operation_id, owner)).fetchone()
        if not row:
            raise FileNotFoundError('Operation not found.')
        if row['status'] not in {'failed', 'cancelled'}:
            raise FileExistsError('Only failed or cancelled operations can be retried.')
        if c.execute("select 1 from label_studio_operations where project_id=%s and status in ('queued','running')", (row['project_id'],)).fetchone():
            raise FileExistsError('This project already has an active operation.')
        if row['kind'] == 'import' and (not row['upload_path'] or not Path(row['upload_path']).is_file()):
            raise ValueError('The ZIP expired. Upload it again.')
        c.execute("update label_studio_operations set status='queued',cancel_requested=false,error=null,error_code=null,error_details='{}',rq_job_id=null,updated_at=now() where id=%s", (operation_id,))
    enqueue(operation_id)
    return operation(owner, operation_id)


def cancel(owner, operation_id):
    with registry._connect() as c:
        c.execute("update label_studio_operations set cancel_requested=true,status=case when status='queued' then 'cancelled' else status end,updated_at=now() where id=%s and owner_user_id=%s and status in ('queued','running')", (operation_id, owner))
    return operation(owner, operation_id)


def progress(operation_id, processed, total, skipped=0):
    with registry._connect() as c:
        row = c.execute('update label_studio_operations set processed=%s,total=%s,skipped=%s,heartbeat_at=now(),updated_at=now() where id=%s returning cancel_requested', (processed, total, skipped, operation_id)).fetchone()
    if not row or row['cancel_requested']:
        raise InterruptedError('Operation cancelled. Already imported images remain available.')


def enough_disk(path, required):
    if shutil.disk_usage(path).free < required + 2 * 1024**3:
        raise ValueError('Insufficient free disk space. Free space or increase storage before retrying.')


def import_images(op, project, stage):
    upload = Path(op['upload_path'])
    root = stage / 'extracted'
    with zipfile.ZipFile(upload) as archive:
        expanded = sum(info.file_size for info in archive.infolist())
        enough_disk(stage, expanded * 2)
        safe_extract_zip(archive, root)
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    skipped, imported, duplicates, errors = 0, 0, 0, []
    for index, path in enumerate(files):
        progress(op['id'], index, len(files), skipped)
        normalized = stage / 'normalized.png'
        try:
            with Image.open(path) as source:
                if source.width * source.height > 25_000_000:
                    raise ValueError('Image exceeds 25 million pixels.')
                ImageOps.exif_transpose(source).convert('RGB').save(normalized, format='PNG')
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            skipped += 1
            if len(errors) < 100:
                errors.append({'file': str(path.relative_to(root)), 'reason': str(exc)[:160]})
            continue
        with normalized.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        result = bridge(op['owner_user_id'], 'image-upload', {
            'project': project['ls_project_id'], 'sha256': sha, 'filename': str(path.relative_to(root)),
        }, content=normalized)
        if result['duplicate']:
            duplicates += 1
        else:
            imported += 1
    progress(op['id'], len(files), len(files), skipped)
    if not imported and not duplicates:
        raise ValueError('No usable images were found in the ZIP.')
    return {'imported': imported, 'duplicates': duplicates, 'skipped': skipped, 'errors': errors}


def publish(op, project, stage):
    owner = op['owner_user_id']
    if not op['snapshot_id']:
        snapshot = bridge(owner, 'snapshot', {'project': project['ls_project_id'], 'operation': str(op['id'])})
        op['snapshot_id'] = snapshot['snapshot']
        with registry._connect() as c:
            c.execute('update label_studio_operations set snapshot_id=%s where id=%s', (op['snapshot_id'], op['id']))
    first = bridge(owner, 'snapshot-page', {'snapshot': op['snapshot_id'], 'offset': 0})
    config, total = first['config'], first['total']
    if op['version'] is None:
        with registry._connect() as c:
            current = c.execute('update label_studio_projects set reserved_version=greatest(reserved_version,published_version)+1 where id=%s returning reserved_version', (project['id'],)).fetchone()
            op['version'] = current['reserved_version']
            c.execute('update label_studio_operations set version=%s where id=%s', (op['version'], op['id']))
    slug = f"{safe_dataset_name(config['name'])[:70]}-{str(project['id'])[:8]}-v{op['version']}"
    target = owner_dataset_path(DATASET_DIR, owner, slug)
    with registry.dataset_guard(owner, slug) as c:
        existing = registry.get_dataset(owner, slug, c)
        if existing:
            if existing['metadata'].get('label_studio_operation') != str(op['id']):
                raise FileExistsError('The dataset name is already in use.')
            return {'datasetId': str(existing['id']), 'datasetName': slug, 'version': op['version']}
    output = stage / 'dataset'
    output.mkdir()
    exporter = SnapshotExporter(output, owner, op['snapshot_id'], config)
    for offset in range(total):
        progress(op['id'], offset, total)
        page = first if offset == 0 else bridge(owner, 'snapshot-page', {'snapshot': op['snapshot_id'], 'offset': offset})
        if len(page['items']) != 1:
            raise ValueError('Snapshot is incomplete.')
        item = page['items'][0]
        enough_disk(stage, item['width'] * item['height'] * 10)
        exporter.write_image(item)
    schema = exporter.finish()
    metadata = inspect_dataset(output)
    validate_dataset_for_upload(output, metadata)
    metadata['label_schema'] = schema
    metadata['label_studio_operation'] = str(op['id'])
    manifest = {'created_by': owner, 'label_schema': schema, 'source_label_studio_project': project['ls_project_id'],
                'snapshot': op['snapshot_id'], 'label_studio_operation': str(op['id']), 'version': op['version'], 'revision': config.get('revision'),
                'split': {'ratios': config['ratios'], 'seed': config['seed']},
                'workflow': dataset_workflow_metadata(metadata, get_catalog())}
    (output / '.ailab_dataset.json').write_text(json.dumps(manifest))
    progress(op['id'], total, total)
    if op.get('preview_only'):
        preview = contained_path(DATASET_DIR / '.label-studio-previews', str(op['id']))
        preview.parent.mkdir(parents=True, exist_ok=True)
        if not preview.exists():
            output.replace(preview)
        return {'datasetName': slug, 'version': op['version'], 'preview': manifest['workflow'],
                'images': total, 'classes': [c['name'] for c in config['classes']], 'accepted': False}
    with registry.dataset_guard(owner, slug) as c:
        if target.exists():
            # A crash may occur after rename but before the DB transaction commits.
            prior = json.loads((target / '.ailab_dataset.json').read_text())
            if prior.get('label_studio_operation') != str(op['id']):
                raise FileExistsError('Dataset destination already exists.')
            metadata = inspect_dataset(target)
            validate_dataset_for_upload(target, metadata)
            metadata.update(label_schema=schema, label_studio_operation=str(op['id']))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            output.replace(target)
        record = registry.upsert_dataset(owner, None, slug, target, metadata, c)
        c.execute('update label_studio_projects set published_version=greatest(published_version,%s),published_revision=%s where id=%s', (op['version'], config.get('revision'), project['id']))
    return {'datasetId': str(record['id']), 'datasetName': slug, 'version': op['version']}


def accept_preview(owner, operation_id, discard=False):
    with registry._connect() as c:
        op = c.execute('select * from label_studio_operations where id=%s and owner_user_id=%s for update', (operation_id, owner)).fetchone()
        if not op or not op['preview_only']:
            raise FileNotFoundError('Preview not found.')
        if op['result'].get('accepted'):
            return op['result']
        if op['status'] != 'completed':
            raise ValueError('Preview is not ready.')
        source = contained_path(DATASET_DIR / '.label-studio-previews', str(op['id']))
        if discard:
            shutil.rmtree(source, ignore_errors=True)
            c.execute("update label_studio_operations set status='cancelled',updated_at=now() where id=%s", (op['id'],))
            return {'discarded': True}
        slug = op['result']['datasetName']
        target = owner_dataset_path(DATASET_DIR, owner, slug)
        with registry.dataset_guard(owner, slug) as dc:
            path = target if target.exists() else source
            if not path.exists():
                raise ValueError('Preview expired. Send the project again from Label Studio.')
            manifest = json.loads((path / '.ailab_dataset.json').read_text())
            if manifest.get('label_studio_operation') != str(op['id']):
                raise FileExistsError('Another version was imported. Send the project again.')
            metadata = inspect_dataset(path)
            validate_dataset_for_upload(path, metadata)
            metadata.update(label_schema=manifest['label_schema'], label_studio_operation=str(op['id']))
            if path == source:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
            record = registry.upsert_dataset(owner, None, slug, target, metadata, dc)
        result = {**op['result'], 'accepted': True, 'datasetId': str(record['id'])}
        c.execute('update label_studio_projects set published_version=greatest(published_version,%s),published_revision=%s where id=%s', (op['version'], manifest.get('revision'), op['project_id']))
        c.execute('update label_studio_operations set result=%s::jsonb,updated_at=now() where id=%s', (json.dumps(result), op['id']))
        return result


def run(operation_id):
    # Hold a database lock for the entire job, including filesystem publication.
    # Redis loss must not allow recovery to start another copy of this operation.
    with registry._connect() as lock:
        locked = lock.execute('select pg_try_advisory_xact_lock(hashtextextended(%s, 0)) as acquired', ('label-studio:' + str(operation_id),)).fetchone()
        if locked['acquired']:
            _run(operation_id)


def _run(operation_id):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with registry._connect() as c:
        row = c.execute("update label_studio_operations set status='running',attempt=attempt+1,heartbeat_at=now(),updated_at=now(),error=null where id=%s and status='queued' and not cancel_requested returning *", (operation_id,)).fetchone()
        if not row:
            return
        op = dict(row)
        project = c.execute('select * from label_studio_projects where id=%s and owner_user_id=%s and deleted_at is null', (op['project_id'], op['owner_user_id'])).fetchone()
    stage = contained_path(WORK_DIR, str(op['id']))
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        if not project:
            raise FileNotFoundError('Project was removed.')
        result = import_images(op, project, stage) if op['kind'] == 'import' else publish(op, project, stage)
        with registry._connect() as c:
            c.execute("update label_studio_operations set status='completed',result=%s::jsonb,updated_at=now() where id=%s", (json.dumps(result), operation_id))
        if op['upload_path']:
            Path(op['upload_path']).unlink(missing_ok=True)
    except InterruptedError as exc:
        with registry._connect() as c:
            c.execute("update label_studio_operations set status='cancelled',error=%s,updated_at=now() where id=%s", (str(exc), operation_id))
    except Exception as exc:
        logger.exception('Label Studio operation %s failed', operation_id)
        message = operation_error(exc)
        info = error_info(exc, message)
        with registry._connect() as c:
            c.execute("update label_studio_operations set status='failed',error=%s,error_code=%s,error_details=%s::jsonb,updated_at=now() where id=%s", (message, info['code'], json.dumps(info['details']), operation_id))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def recover():
    redis = Redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=5)
    redis.ping()
    with registry._connect() as c:
        rows = c.execute("select * from label_studio_operations where status in ('queued','running')").fetchall()
    for row in rows:
        alive = False
        if row['rq_job_id']:
            try:
                job = Job.fetch(row['rq_job_id'], connection=redis)
                status = str(job.get_status()).split('.')[-1].lower()
                alive = status in {'queued', 'scheduled', 'deferred'}
                if status == 'started':
                    alive = any(worker.get_current_job_id() == job.id for worker in Worker.all(connection=redis))
            except Exception:
                pass
        # A running operation holds the PostgreSQL lock. RQ heartbeats may survive
        # a killed container for minutes, so they cannot prove it is still alive.
        if alive and row['status'] == 'queued':
            continue
        with registry._connect() as c:
            locked = c.execute('select pg_try_advisory_xact_lock(hashtextextended(%s, 0)) as acquired', ('label-studio:' + str(row['id']),)).fetchone()
            if not locked['acquired']:
                continue
            if row['status'] == 'running' and row['attempt'] >= 3 and not row['cancel_requested']:
                c.execute("update label_studio_operations set status='failed',error='Worker stopped repeatedly. Check memory/storage limits before retrying.',updated_at=now() where id=%s and status='running'", (row['id'],))
                continue
            c.execute("update label_studio_operations set status=case when cancel_requested then 'cancelled' else 'queued' end,updated_at=now() where id=%s and status in ('running','queued')", (row['id'],))
        if not row['cancel_requested']:
            enqueue(row['id'])


def cleanup():
    with registry._connect() as c:
        expired = c.execute("select id from label_studio_operations where preview_only and status='completed' and not coalesce((result->>'accepted')::boolean,false) and updated_at < now() - interval '7 days' for update skip locked").fetchall()
        for row in expired:
            shutil.rmtree(contained_path(DATASET_DIR / '.label-studio-previews', str(row['id'])), ignore_errors=True)
            c.execute("update label_studio_operations set status='cancelled',error='Preview expired. Send the project again.',updated_at=now() where id=%s", (row['id'],))
    with registry._connect() as c:
        rows = c.execute("select id from label_studio_operations where (upload_path is not null or snapshot_id is not null) and (status='completed' or (status in ('failed','cancelled') and updated_at < now() - interval '7 days'))").fetchall()
    for row in rows:
        with registry._connect() as c:
            row = c.execute("select * from label_studio_operations where id=%s and (status='completed' or (status in ('failed','cancelled') and updated_at < now() - interval '7 days')) for update skip locked", (row['id'],)).fetchone()
            if not row:
                continue
            if row['snapshot_id']:
                bridge(row['owner_user_id'], 'release-snapshot', {'snapshot': row['snapshot_id']})
                c.execute('update label_studio_operations set snapshot_id=null where id=%s', (row['id'],))
            if row['upload_path']:
                path = contained_path(DATASET_DIR / '.label-studio-uploads', row['upload_path'])
                path.unlink(missing_ok=True)
                c.execute('update label_studio_operations set upload_path=null where id=%s', (row['id'],))
            shutil.rmtree(contained_path(WORK_DIR, str(row['id'])), ignore_errors=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == 'recover':
        recover()
    else:
        next_cleanup = 0
        while True:
            try:
                recover()
                redis = Redis.from_url(REDIS_URL)
                Worker([Queue(QUEUE, connection=redis)], connection=redis).work(burst=True, max_jobs=1, logging_level='WARNING')
                if time.monotonic() >= next_cleanup:
                    cleanup()
                    next_cleanup = time.monotonic() + 3600
            except Exception:
                logger.exception('Label Studio queue unavailable; retrying.')
            time.sleep(5)
