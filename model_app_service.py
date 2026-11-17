"""Session-only Model Apps; the model owner pays, the requester owns their results."""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import quote

from api_errors import PublicError
import compute_repository as compute
from label_studio_api import owner
import model_api_service as models
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import RUNS_DIR


def available():
    if not models.enabled():
        raise PublicError('MODEL_APPS_UNAVAILABLE', 'Model Apps are temporarily unavailable.', status=503)
    compute.require_enabled()


def app_path(row):
    return f"/apps/{quote(row['owner_user_id'], safe='')}/{row['slug']}"


def lock_owner(connection, subject):
    connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
    quota = connection.execute('select accepting_jobs from compute_user_quotas where owner_user_id=%s', (subject,)).fetchone()
    if quota and not quota['accepting_jobs']:
        raise PublicError('USER_RESOURCES_DELETING', 'This account is undergoing resource cleanup.', status=409)


def owned(connection, subject, app_id):
    row = connection.execute('select * from model_apps where id=%s and owner_user_id=%s for update', (app_id, subject)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Model App not found.', status=404)
    return row


def add_release(connection, app, run_slug):
    run = connection.execute('''select * from training_tasks where run_slug=%s and owner_user_id=%s
        for update''', (run_slug, app['owner_user_id'])).fetchone()
    if not run:
        raise PublicError('NOT_FOUND', 'Run not found.', status=404)
    if run['status'] != 'completed':
        raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be published.', status=409)
    previous = connection.execute('select task_type,model_type from model_app_releases where app_id=%s limit 1', (app['id'],)).fetchone()
    if previous and (previous['task_type'] != run['task_type'] or
                     (previous['model_type'] == 'mask_rcnn') != (run['model_type'] == 'mask_rcnn')):
        raise PublicError('APP_TASK_MISMATCH', 'Select a run with the same task type as this app.', status=409)
    try:
        path, digest = models.checkpoint(run['run_slug'], run['model_type'])
        config = json.loads(contained_path(RUNS_DIR, run_slug, 'job_config.json').read_text())
    except (OSError, ValueError) as exc:
        raise PublicError('CHECKPOINT_UNAVAILABLE', 'The saved model or configuration is unavailable.', status=409) from exc
    existing = connection.execute('select id from model_app_releases where app_id=%s and run_id=%s and checkpoint_sha256=%s',
                                  (app['id'], run['id'], digest)).fetchone()
    if existing:
        raise PublicError('APP_RELEASE_EXISTS', 'This model release is already in the app.', status=409)
    if app['next_version'] > 100:
        raise PublicError('APP_RELEASE_LIMIT', 'This app has reached its release limit.', status=409)
    # Checkpoints carry the class mapping and inference preprocessing; retain only estimator inputs here.
    connection.execute('''insert into model_app_releases(app_id,version,run_id,checkpoint_path,checkpoint_sha256,
        model_type,task_type,config) values(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)''',
        (app['id'], app['next_version'], run['id'], path, digest, run['model_type'], run['task_type'],
         json.dumps({'extra_args': config.get('extra_args', {})})))
    connection.execute('update model_apps set next_version=next_version+1,updated_at=now() where id=%s', (app['id'],))


def describe(connection, row, *, managing=False, version=None):
    releases = connection.execute('''select r.*,t.run_slug from model_app_releases r
        join training_tasks t on t.id=r.run_id where r.app_id=%s order by r.version desc''', (row['id'],)).fetchall()
    selected = next((r for r in releases if r['version'] == (version or row['active_version'])), None)
    if selected is None and managing and releases:
        selected = releases[0]
    result = {'id': str(row['id']), 'name': row['name'], 'slug': row['slug'], 'description': row['description'],
              'path': app_path(row), 'published': row['published'], 'activeVersion': row['active_version'],
              'version': selected['version'] if selected else None,
              'taskType': ({'mask_rcnn': 'instance_segmentation', 'deeplabv3plus': 'semantic_segmentation'}.get(selected['model_type'], selected['task_type'])) if selected else None,
              'modelType': selected['model_type'] if selected else None, 'isOwner': managing}
    if managing:
        result['releases'] = [{'version': r['version'], 'runSlug': r['run_slug'], 'testedAt': r['tested_at'],
                               'checkpointSha256': r['checkpoint_sha256'], 'createdAt': r['created_at']} for r in releases]
        result['usage'] = dict(connection.execute('''select count(*) as requests,
            count(*) filter(where status='completed') as completed,
            count(*) filter(where status='failed') as failed,
            count(*) filter(where status=any(%s)) as pending,
            round(avg(extract(epoch from started_at-created_at))::numeric,2) as "averageQueueSeconds"
            from compute_jobs where model_app_id=%s''', (['queued', *compute.ACTIVE], row['id'])).fetchone())
    return result


def resolve(subject, owner_id, slug, version=None, connection=None, *, lock=False):
    own_connection = connection is None
    c = connection or registry._connect()
    try:
        row = c.execute('''select a.* from model_apps a join "user" u on u.id=a.owner_user_id
            where a.owner_user_id=%s and a.slug=%s and u."emailVerified" and not coalesce(u.banned,false)'''
            + (' for share of a,u' if lock else ''), (owner_id, slug)).fetchone()
        if not row or (subject != owner_id and (not row['published'] or version is not None)):
            raise PublicError('NOT_FOUND', 'Model App not found or not published.', status=404)
        selected = version if version is not None else row['active_version']
        if not selected:
            raise PublicError('APP_RELEASE_REQUIRED', 'Select a release to preview before publishing.', status=409)
        release = c.execute('''select r.*,t.run_slug,t.status as run_status from model_app_releases r
            join training_tasks t on t.id=r.run_id where r.app_id=%s and r.version=%s
            and t.owner_user_id=%s''', (row['id'], selected, owner_id)).fetchone()
        if not release or release['run_status'] != 'completed':
            raise PublicError('NOT_FOUND', 'Model release not found.', status=404)
        return dict(row), dict(release)
    finally:
        if own_connection:
            c.close()


def upload_context(request):
    available()
    subject = owner(request)['id']
    match = re.fullmatch(r'/api/apps/([^/]+)/([^/]+)/predict', request.url.path)
    if not match:
        raise PublicError('NOT_FOUND', 'Model App not found.', status=404)
    raw = request.query_params.get('version')
    if raw is not None and (not raw.isascii() or not raw.isdigit() or len(raw) > 3 or int(raw) < 1):
        raise PublicError('INVALID_RELEASE', 'Invalid model release.')
    app, release = resolve(subject, *match.groups(), int(raw) if raw is not None else None)
    # Shared owner budget with API traffic; visitors also have their own submission limit.
    models.enforce_rate_limit({'owner_user_id': app['owner_user_id']}, True)
    if subject != app['owner_user_id']:
        models.enforce_rate_limit({'owner_user_id': 'app-visitor:' + subject}, True)
    return {'owner_user_id': app['owner_user_id'], 'requester_id': subject, 'app_id': app['id'],
            'release_id': release['id'], 'version': release['version'], 'preview': raw is not None,
            'path': app_path(app), 'name': app['name'], 'checkpoint_path': release['checkpoint_path'],
            'checkpoint_sha256': release['checkpoint_sha256'], 'release': release}


def read_limit(request):
    subject = owner(request)['id']
    models.enforce_rate_limit({'key_id': 'app-session:' + subject}, False)


def check_admission(c, context, owner_id, run_id, payload):
    app, release = resolve(context['requester_id'], owner_id, context['path'].rsplit('/', 1)[1],
                           context['version'] if context['preview'] else None, connection=c, lock=True)
    active = c.execute('''select id from "user" where id=%s and "emailVerified" and not coalesce(banned,false)
        for share''', (context['requester_id'],)).fetchone()
    if not active or str(app['id']) != str(context['app_id']) or str(release['id']) != str(context['release_id']):
        raise PublicError('APP_RELEASE_CHANGED', 'This app changed. Refresh it before sending another image.', status=409)
    if str(release['run_id']) != str(run_id) or release['checkpoint_sha256'] != payload.get('checkpoint_sha256'):
        raise PublicError('APP_RELEASE_CHANGED', 'The selected model release changed.', status=409)
    c.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('app-visitor:' + context['requester_id']),))
    quota = c.execute('select accepting_jobs from compute_user_quotas where owner_user_id=%s', (context['requester_id'],)).fetchone()
    if quota and not quota['accepting_jobs']:
        raise PublicError('USER_RESOURCES_DELETING', 'This account is undergoing resource cleanup.', status=409)


def request_key(request, context):
    key = request.headers.get('idempotency-key', '')
    if not key or len(key) > 128 or any(ord(char) < 32 for char in key):
        raise PublicError('IDEMPOTENCY_REQUIRED', 'A valid request key is required.')
    digest = hashlib.sha256(f"{context['requester_id']}:{context['app_id']}:{key}".encode()).hexdigest()
    return 'app:' + digest


def job_for(subject, job_id):
    with registry._connect() as c:
        row = c.execute('''select j.*,g.name as gpu_name from compute_jobs j
            left join gpu_devices g on g.uuid=j.assigned_gpu_uuid
            where j.id=%s and j.is_model_app and j.app_requester_user_id=%s''', (job_id, subject)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Prediction not found.', status=404)
    compute.require_unexpired_result(row)
    return dict(row)


def public_job(row):
    result = compute.public_job(row)
    result.pop('assignedGpu', None)
    result.pop('requestedExecution', None)
    result['error'] = None  # Worker diagnostics can contain the owner's filesystem paths.
    result['version'] = row['payload']['app_version']
    result['appId'] = str(row['model_app_id']) if row['model_app_id'] else row['payload'].get('app_id')
    return result
