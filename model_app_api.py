from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from api_errors import PublicError
import compute_repository as compute
from inference_service import MAX_INFERENCE_IMAGE_BYTES
from label_studio_api import owner
import model_api_service as models
import model_app_service as service
from prediction_submission import submit_single
from resource_repository import resource_repository as registry

management_router = APIRouter(prefix='/api/model-apps')
router = APIRouter(prefix='/api/apps')


class CreateApp(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{0,62}$')
    description: str = Field(default='', max_length=1000)
    runSlug: str = Field(min_length=1, max_length=128)


class UpdateApp(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    published: bool | None = None
    version: int | None = Field(default=None, ge=1, le=100)


class CreateRelease(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    runSlug: str = Field(min_length=1, max_length=128)


@management_router.get('')
def list_apps(request: Request):
    service.available()
    subject = owner(request)['id']
    with registry._connect() as c:
        rows = c.execute('select * from model_apps where owner_user_id=%s order by created_at desc', (subject,)).fetchall()
        return {'apps': [service.describe(c, row, managing=True) for row in rows]}


@management_router.post('', status_code=201)
def create_app(body: CreateApp, request: Request):
    service.available()
    subject = owner(request)['id']
    with registry._connect() as c:
        service.lock_owner(c, subject)
        if c.execute('select 1 from model_apps where owner_user_id=%s and slug=%s', (subject, body.slug)).fetchone():
            raise PublicError('APP_SLUG_EXISTS', 'This address is already in use. Choose another name.', status=409)
        if c.execute('select count(*) as n from model_apps where owner_user_id=%s', (subject,)).fetchone()['n'] >= 50:
            raise PublicError('APP_LIMIT', 'This account has reached its Model App limit.', status=409)
        row = c.execute('''insert into model_apps(owner_user_id,name,slug,description) values(%s,%s,%s,%s)
            returning *''', (subject, body.name, body.slug, body.description)).fetchone()
        service.add_release(c, row, body.runSlug)
        return service.describe(c, row, managing=True)


@management_router.patch('/{app_id}')
def update_app(app_id: uuid.UUID, body: UpdateApp, request: Request):
    service.available()
    subject = owner(request)['id']
    with registry._connect() as c:
        service.lock_owner(c, subject)
        app = service.owned(c, subject, app_id)
        version = body.version or app['active_version']
        published = app['published'] if body.published is None else body.published
        if body.version is not None or (published and not app['published']):
            if not version:
                raise PublicError('APP_RELEASE_REQUIRED', 'Choose a tested model release.', status=409)
            _, release = service.resolve(subject, subject, app['slug'], version, connection=c)
            if not release['tested_at']:
                raise PublicError('APP_TEST_REQUIRED', 'Complete a preview prediction before publishing this release.', status=409)
            models.verify_checkpoint(release['run_slug'], release['model_type'], release['checkpoint_path'], release['checkpoint_sha256'])
        row = c.execute('''update model_apps set name=%s,description=%s,published=%s,active_version=%s,
            updated_at=now() where id=%s returning *''',
            (body.name if body.name is not None else app['name'], body.description if body.description is not None else app['description'],
             published, version, app_id)).fetchone()
        return service.describe(c, row, managing=True)


@management_router.post('/{app_id}/releases', status_code=201)
def add_release(app_id: uuid.UUID, body: CreateRelease, request: Request):
    service.available()
    subject = owner(request)['id']
    with registry._connect() as c:
        service.lock_owner(c, subject)
        app = service.owned(c, subject, app_id)
        service.add_release(c, app, body.runSlug)
        return service.describe(c, app, managing=True)


@management_router.delete('/{app_id}')
def delete_app(app_id: uuid.UUID, request: Request):
    service.available()
    subject = owner(request)['id']
    with registry._connect() as c:
        service.lock_owner(c, subject)
        app = service.owned(c, subject, app_id)
        if app['published'] or c.execute('select 1 from compute_jobs where model_app_id=%s and status=any(%s) limit 1',
                                        (app_id, ['queued', *compute.ACTIVE])).fetchone():
            raise PublicError('APP_IN_USE', 'Unpublish this app and wait for its predictions to finish before deleting.', status=409)
        c.execute('delete from model_apps where id=%s', (app_id,))
    return {'deleted': True}


@router.get('/jobs/{job_id}')
def job(job_id: uuid.UUID, request: Request):
    return service.public_job(service.job_for(owner(request)['id'], job_id))


@router.get('/jobs/{job_id}/result')
def result(job_id: uuid.UUID, request: Request):
    subject = owner(request)['id']
    row = service.job_for(subject, job_id)
    data = compute.read_result(row['owner_user_id'], job_id, requester_id=subject)
    # Return predictions, not internal checkpoint paths, owner run slugs or runtime diagnostics.
    result = {key: data[key] for key in ('taskType', 'predictions', 'top', 'classes', 'image', 'timingMs',
                                        'count', 'counts', 'threshold', 'segmentation') if key in data}
    result['version'] = row['payload']['app_version']
    if result.get('segmentation', {}).get('overlay'):
        result['segmentation']['overlay'] = f'/api/apps/jobs/{job_id}/files/overlay'
    return result


@router.get('/jobs/{job_id}/files/{filename}')
def file(job_id: uuid.UUID, filename: str, request: Request):
    from PIL import Image
    row = service.job_for(owner(request)['id'], job_id)
    if filename not in {'input', 'overlay'}:
        raise PublicError('NOT_FOUND', 'File not found.', status=404)
    if filename == 'overlay' and row['status'] != 'completed':
        raise PublicError('RESULT_NOT_READY', 'Prediction is not ready.', status=409)
    path = compute.job_directory(job_id) / ('overlay.png' if filename == 'overlay' else 'input')
    if not path.is_file():
        raise PublicError('NOT_FOUND', 'Image not found.', status=404)
    with Image.open(path) as image:
        mime = Image.MIME.get(image.format, 'application/octet-stream')
    return FileResponse(path, media_type=mime, headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id: uuid.UUID, request: Request):
    subject = owner(request)['id']
    row = service.job_for(subject, job_id)
    compute.cancel_job(row['owner_user_id'], job_id, requester_id=subject)
    return service.public_job(service.job_for(subject, job_id))


@router.get('/{owner_id}/{slug}')
def shared_app(owner_id: str, slug: str, request: Request, version: int | None = Query(default=None, ge=1, le=100), job: uuid.UUID | None = None):
    service.available()
    subject = owner(request)['id']
    with registry._connect() as c:
        if job is not None:
            previous = service.job_for(subject, job)
            app = c.execute('select * from model_apps where id=%s and owner_user_id=%s and slug=%s',
                            (previous['model_app_id'], owner_id, slug)).fetchone()
            if not app:
                if previous['payload'].get('app_path') != service.app_path({'owner_user_id': owner_id, 'slug': slug}):
                    raise PublicError('NOT_FOUND', 'Model App not found.', status=404)
                return {'id': previous['payload']['app_id'], 'name': previous['payload']['app_name'],
                        'slug': slug, 'description': '', 'path': previous['payload']['app_path'], 'published': False,
                        'activeVersion': None, 'version': previous['payload']['app_version'], 'isOwner': False,
                        'taskType': previous['payload']['task_type'], 'modelType': previous['payload']['model_type'],
                        'jobs': [service.public_job(previous)]}
            selected = version if subject == owner_id and version else (app['active_version'] or previous['payload']['app_version'])
        else:
            app, release = service.resolve(subject, owner_id, slug, version)
            selected = release['version']
        result = service.describe(c, app, managing=subject == owner_id, version=selected)
        jobs = c.execute('''select * from compute_jobs where model_app_id=%s and app_requester_user_id=%s
            and cleaned_at is null and (finished_at is null or finished_at>now()-interval '24 hours')
            order by created_at desc limit 10''', (app['id'], subject)).fetchall()
        result['jobs'] = [service.public_job(row) for row in jobs]
        return result


@router.post('/{owner_id}/{slug}/predict', status_code=202)
async def predict(owner_id: str, slug: str, request: Request, file: UploadFile = File(...),
                  threshold: float | None = Form(None)):
    context = request.state.model_app_context
    key = service.request_key(request, context)
    release = context['release']
    payload = await file.read(MAX_INFERENCE_IMAGE_BYTES + 1)
    if len(payload) > MAX_INFERENCE_IMAGE_BYTES:
        raise PublicError('IMAGE_TOO_LARGE', 'The image exceeds the 10 MiB limit.', status=413)
    run = {'id': release['run_id'], 'run_slug': release['run_slug'], 'status': release['run_status'],
           'model_type': release['model_type'], 'task_type': release['task_type']}
    await run_in_threadpool(models.verify_checkpoint, run['run_slug'], run['model_type'],
                            release['checkpoint_path'], release['checkpoint_sha256'])
    row = await submit_single(context['owner_user_id'], run, release['config'], payload, file.filename or 'image',
                              {'mode': 'auto'}, threshold, key, app_context=context)
    return service.public_job(row)
