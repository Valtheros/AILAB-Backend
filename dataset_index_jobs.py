"""Durable CPU-only dataset indexing using the existing data worker."""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from redis import Redis
from rq import Queue
from rq.job import Job

from api_errors import PublicError
from dataset_indexer import add_stats, index_image, source_images
from dataset_versioning import verify_source
from dataset_storage import registered_storage_path
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import DATASET_DIR, REDIS_URL

log = logging.getLogger(__name__)
QUEUE = 'dataset_index'


def public(row):
    return {'status': row['status'], 'total': row['total'], 'processed': row['processed'],
            'skipped': row['skipped'], 'error': row['error_message'], 'stats': row['stats'],
            'ready': row['active_generation'] is not None}


def request_index(subject, dataset_id):
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
        dataset = connection.execute('''select d.id from datasets d join "user" u on u.id=d.owner_user_id
            where d.id=%s and d.owner_user_id=%s and d.status='active' and u."emailVerified"
            and not coalesce(u.banned,false) for update of d,u''', (dataset_id, subject)).fetchone()
        if not dataset:
            raise PublicError('NOT_FOUND', 'Dataset not found.', status=404)
        row = connection.execute('select * from dataset_indexes where dataset_version_id=%s for update', (dataset_id,)).fetchone()
        if not row:
            row = connection.execute('''insert into dataset_indexes(dataset_version_id,owner_user_id,working_generation)
                values(%s,%s,%s) returning *''', (dataset_id, subject, uuid.uuid4())).fetchone()
        elif row['status'] == 'failed':
            row = connection.execute('''update dataset_indexes set status='queued',working_generation=%s,rq_job_id=null,
                total=0,processed=0,skipped=0,error_message=null,updated_at=now()
                where dataset_version_id=%s returning *''', (uuid.uuid4(), dataset_id)).fetchone()
        connection.execute('''insert into dataset_index_generations(dataset_version_id,generation)
            values(%s,%s) on conflict do nothing''', (dataset_id, row['working_generation']))
    if row['status'] == 'queued':
        enqueue(dataset_id, row['working_generation'])
    return public(row)


def enqueue(dataset_id, generation):
    try:
        redis = Redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=5)
        queue = Queue(QUEUE, connection=redis)
        job_id = f'index-{dataset_id}-{uuid.uuid4().hex[:8]}'
        with registry._connect() as connection:
            row = connection.execute('''update dataset_indexes set rq_job_id=%s where dataset_version_id=%s
                and working_generation=%s and status='queued' and rq_job_id is null returning dataset_version_id''',
                (job_id, dataset_id, generation)).fetchone()
        if row:
            queue.enqueue('dataset_index_jobs.run', str(dataset_id), str(generation), job_id=job_id,
                          job_timeout=21600, result_ttl=3600, failure_ttl=86400)
    except Exception:
        log.warning('Dataset index will wait for queue recovery.', exc_info=True)


def recover():
    redis = Redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=5)
    redis.ping()
    with registry._connect() as connection:
        rows = connection.execute("select * from dataset_indexes where status in ('queued','running')").fetchall()
    for candidate in rows:
        alive = False
        if candidate['rq_job_id']:
            try:
                job = Job.fetch(candidate['rq_job_id'], connection=redis)
                state = str(job.get_status()).split('.')[-1].lower()
                alive = state in {'queued', 'scheduled', 'deferred'}
            except Exception:
                pass
        if candidate['status'] == 'queued' and alive:
            continue
        with registry._connect() as connection:
            locked = connection.execute('select pg_try_advisory_xact_lock(hashtextextended(%s,0)) as acquired',
                                        ('dataset-index:' + str(candidate['dataset_version_id']),)).fetchone()
            if not locked['acquired']:
                continue
            row = connection.execute('''update dataset_indexes set status='queued',rq_job_id=null,
                working_generation=case when status='running' then gen_random_uuid() else working_generation end,
                processed=0,skipped=0,updated_at=now() where dataset_version_id=%s and status in ('queued','running') returning *''',
                (candidate['dataset_version_id'],)).fetchone()
            if row:
                connection.execute('''insert into dataset_index_generations(dataset_version_id,generation)
                    values(%s,%s) on conflict do nothing''', (row['dataset_version_id'], row['working_generation']))
        if row:
            enqueue(row['dataset_version_id'], row['working_generation'])


def run(dataset_id, generation):
    with registry._connect() as lock:
        acquired = lock.execute('select pg_try_advisory_xact_lock(hashtextextended(%s,0)) as acquired',
                                ('dataset-index:' + str(dataset_id),)).fetchone()['acquired']
        if acquired:
            _run(dataset_id, generation)


def _run(dataset_id, generation):
    with registry._connect() as connection:
        row = connection.execute('''update dataset_indexes set status='running',updated_at=now() where dataset_version_id=%s
            and working_generation=%s and status='queued' returning *''', (dataset_id, generation)).fetchone()
        if not row:
            return
        dataset = connection.execute("select * from datasets where id=%s and owner_user_id=%s and status='active'",
                                     (dataset_id, row['owner_user_id'])).fetchone()
    try:
        if not dataset:
            raise ValueError('Dataset is no longer available.')
        root = registered_storage_path(DATASET_DIR, dataset['storage_path'])
        verify_source(root, dataset['content_sha256'])
        generation_id = str(uuid.UUID(str(generation)))
        cache = contained_path(root, '.ailab_index', generation_id)
        cache.mkdir(parents=True, exist_ok=False)
        with registry._connect() as connection:
            connection.execute('''update dataset_index_generations set cache_created=true
                where dataset_version_id=%s and generation=%s''', (dataset_id, generation))
        items = source_images(root, dataset['metadata'] or {})
        if not items:
            raise ValueError('No images were found in supported dataset splits.')
        with registry._connect() as connection:
            connection.execute('update dataset_indexes set total=%s,updated_at=now() where dataset_version_id=%s and working_generation=%s',
                               (len(items), dataset_id, generation))
        stats, skipped, pending = {}, 0, []
        for index, (split, image, source) in enumerate(items):
            thumbnail = contained_path(cache, f'{index}.jpg')
            mask = contained_path(cache, f'{index}.png')
            try:
                summary = index_image(root, image, source, thumbnail, mask)
                annotation = json.dumps(summary['annotation'], ensure_ascii=False, allow_nan=False)
                if len(annotation.encode('utf-8')) > 10 * 1024 * 1024:
                    raise ValueError('Image annotations are too large.')
            except (OSError, ValueError, SyntaxError) as exc:
                log.warning('Skipping unusable dataset image %s: %s', image.name, exc)
                thumbnail.unlink(missing_ok=True)
                mask.unlink(missing_ok=True)
                skipped += 1
                continue
            add_stats(stats, split, summary)
            pending.append((dataset_id, generation, index, split, image.name[:512], image.resolve().relative_to(root).as_posix(),
                            thumbnail.relative_to(root).as_posix(), summary['mask'].relative_to(root).as_posix() if summary['mask'] else None,
                            summary['width'], summary['height'], summary['pixelSha256'], summary['classes'],
                            summary['hasObjects'], annotation))
            if len(pending) >= 50:
                _flush(dataset_id, generation, index + 1, skipped, pending)
                pending.clear()
        _flush(dataset_id, generation, len(items), skipped, pending)
        if sum(value['images'] for value in stats.values()) == 0:
            raise ValueError('No usable images were found.')
        verify_source(root, dataset['content_sha256'])
        with registry._connect() as connection:
            connection.execute('''update dataset_indexes set status='completed',active_generation=%s,stats=%s::jsonb,
                processed=%s,skipped=%s,error_message=null,updated_at=now() where dataset_version_id=%s and working_generation=%s''',
                (generation, json.dumps(stats, ensure_ascii=False), len(items), skipped, dataset_id, generation))
        try:
            _cleanup_previous(dataset_id, generation, root)
        except Exception:
            log.warning('Dataset cache cleanup will retry: %s', dataset_id, exc_info=True)
    except Exception as exc:
        log.exception('Dataset indexing failed for %s', dataset_id)
        message = str(exc)[:300] if isinstance(exc, (ValueError, FileNotFoundError)) else 'Could not prepare dataset details. Please retry.'
        with registry._connect() as connection:
            connection.execute('''update dataset_indexes set status='failed',error_message=%s,updated_at=now()
                where dataset_version_id=%s and working_generation=%s''', (message, dataset_id, generation))


def _flush(dataset_id, generation, processed, skipped, pending):
    with registry._connect() as connection:
        if pending:
            connection.cursor().executemany('''insert into dataset_index_images(dataset_version_id,generation,ordinal,split,
                filename,source_path,thumbnail_path,mask_path,width,height,pixel_sha256,class_names,has_objects,annotations)
                values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)''', pending)
        connection.execute('''update dataset_indexes set processed=%s,skipped=%s,updated_at=now()
            where dataset_version_id=%s and working_generation=%s and status='running' ''',
            (processed, skipped, dataset_id, generation))


def _cleanup_previous(dataset_id, current_generation, root):
    with registry._connect() as connection:
        generations = connection.execute('''select generation from dataset_index_generations where dataset_version_id=%s
            and generation<>%s and cache_created''', (dataset_id, current_generation)).fetchall()
    for row in generations:
        generation = str(row['generation'])
        cache_root = root / '.ailab_index'
        cache = cache_root / generation
        if cache_root.is_symlink() or cache.is_symlink():
            log.warning('Skipping unsafe dataset cache cleanup: %s', dataset_id)
            continue
        try:
            contained_path(root, '.ailab_index', generation)
            if cache.exists():
                shutil.rmtree(cache)
        except (OSError, ValueError):
            log.warning('Dataset cache cleanup will retry: %s', dataset_id, exc_info=True)
            continue
        with registry._connect() as connection:
            connection.execute('delete from dataset_index_images where dataset_version_id=%s and generation=%s', (dataset_id, generation))
            connection.execute('delete from dataset_index_generations where dataset_version_id=%s and generation=%s', (dataset_id, generation))


def cleanup():
    with registry._connect() as connection:
        rows = connection.execute('''select distinct d.id,d.storage_path,i.active_generation from datasets d
            join dataset_indexes i on i.dataset_version_id=d.id join dataset_index_generations g on g.dataset_version_id=d.id
            where d.status='active' and i.status='completed' and i.active_generation is not null
            and g.cache_created and g.generation<>i.active_generation limit 100''').fetchall()
    for row in rows:
        with registry._connect() as lock:
            acquired = lock.execute('select pg_try_advisory_xact_lock(hashtextextended(%s,0)) as acquired',
                                    ('dataset-index:' + str(row['id']),)).fetchone()['acquired']
            if acquired:
                root = registered_storage_path(DATASET_DIR, row['storage_path'])
                _cleanup_previous(row['id'], row['active_generation'], root)
