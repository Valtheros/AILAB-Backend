from __future__ import annotations

import json
import uuid
import os
import tempfile
import hmac
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

from label_studio_client import BridgeError, bridge
import label_studio_jobs as jobs
from resource_repository import resource_repository as registry
from security_utils import MAX_UPLOAD_BYTES, contained_path
from settings import DATASET_DIR
from api_errors import error_info


class Route(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def handle(request):
            try:
                return await original(request)
            except BridgeError as exc:
                return JSONResponse({'detail': str(exc), **error_info(exc), 'requestId': request.state.request_id}, status_code=exc.status)
            except FileNotFoundError as exc:
                return JSONResponse({'detail': str(exc), **error_info(exc), 'requestId': request.state.request_id}, status_code=404)
            except FileExistsError as exc:
                return JSONResponse({'detail': str(exc), **error_info(exc), 'requestId': request.state.request_id}, status_code=409)
            except ValueError as exc:
                return JSONResponse({'detail': str(exc), **error_info(exc), 'requestId': request.state.request_id}, status_code=400)
        return handle


router = APIRouter(prefix='/api/label-studio', route_class=Route)
internal_router = APIRouter(prefix='/internal/label-studio', route_class=Route)


def sync_project(subject, remote_id):
    item = bridge(subject, 'project', {'project': remote_id})
    with registry._connect() as c:
        c.execute('insert into label_studio_projects(id,owner_user_id,ls_project_id,name,kind) values(%s,%s,%s,%s,%s) on conflict(id) do update set name=excluded.name,kind=excluded.kind',
                  (uuid.UUID(item['external_id']), subject, item['id'], item['name'], item['kind']))
    return uuid.UUID(item['external_id'])


def bridge_owner(request):
    secret = os.environ.get('LABEL_STUDIO_BRIDGE_SECRET', '')
    if not secret or not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + secret):
        raise HTTPException(401, 'Unauthorized bridge request.')
    return owner(request)


@internal_router.post('/projects/{remote_id}/imports', status_code=202)
async def studio_upload(remote_id: int, request: Request):
    user = await run_in_threadpool(bridge_owner, request)
    item = json.loads(request.headers.get('x-label-project', '{}'))
    if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ('external_id', 'name', 'kind')):
        raise HTTPException(400, 'Invalid project metadata.')
    project_id = uuid.UUID(item['external_id'])
    if item.get('id') != remote_id:
        raise HTTPException(400, 'Project mismatch.')
    def register():
        with registry._connect() as c:
            c.execute('insert into label_studio_projects(id,owner_user_id,ls_project_id,name,kind) values(%s,%s,%s,%s,%s) on conflict do nothing', (project_id, user['id'], remote_id, item['name'], item['kind']))
    await run_in_threadpool(register)
    return await upload(project_id, request)


@internal_router.get('/projects/{remote_id}/operations')
def studio_operations(remote_id: int, request: Request):
    user = bridge_owner(request)
    with registry._connect() as c:
        project = c.execute('select id from label_studio_projects where ls_project_id=%s and owner_user_id=%s and deleted_at is null', (remote_id, user['id'])).fetchone()
        if not project:
            return {'operations': []}
        project_id = project['id']
        rows = c.execute('select * from label_studio_operations where project_id=%s and owner_user_id=%s order by created_at desc limit 5', (project_id, user['id'])).fetchall()
    return {'operations': [jobs.public_operation(r) for r in rows]}


@internal_router.post('/projects/{remote_id}/operations/{operation_id}/{action}')
def studio_operation_action(remote_id: int, operation_id: uuid.UUID, action: Literal['retry', 'cancel'], request: Request):
    user = bridge_owner(request)
    with registry._connect() as c:
        row = c.execute('select o.id from label_studio_operations o join label_studio_projects p on p.id=o.project_id where o.id=%s and o.owner_user_id=%s and p.ls_project_id=%s and p.deleted_at is null', (operation_id, user['id'], remote_id)).fetchone()
    if not row:
        raise FileNotFoundError('Operation not found.')
    return {'operation': getattr(jobs, action)(user['id'], operation_id)}


class TransferInput(BaseModel):
    transfer: uuid.UUID


@router.post('/transfers/{remote_id}', status_code=202)
def prepare_transfer(remote_id: int, payload: TransferInput, request: Request):
    user = owner(request)
    project_id = sync_project(user['id'], remote_id)
    return {'operation': jobs.create(user['id'], project_id, 'publish', preview_only=True, operation_id=payload.transfer)}


@router.post('/operations/{operation_id}/accept')
def accept_transfer(operation_id: uuid.UUID, request: Request):
    return jobs.accept_preview(owner(request)['id'], operation_id)


@router.post('/operations/{operation_id}/discard')
def discard_transfer(operation_id: uuid.UUID, request: Request):
    return jobs.accept_preview(owner(request)['id'], operation_id, discard=True)


def owner(request):
    subject = request.headers.get('x-user-id')
    if not subject:
        raise HTTPException(401, 'Authentication required.')
    with registry._connect() as c:
        user = c.execute('select id,email,"emailVerified",banned from "user" where id=%s', (subject,)).fetchone()
    if not user or not user['emailVerified'] or user.get('banned'):
        raise HTTPException(403, 'An active verified AILAB account is required.')
    return user


def project_for(subject, project_id):
    with registry._connect() as c:
        row = c.execute('select * from label_studio_projects where owner_user_id=%s and id=%s and deleted_at is null', (subject, project_id)).fetchone()
    if not row:
        raise FileNotFoundError('Project not found.')
    return dict(row)


@router.get('/account')
def account(request: Request):
    user = owner(request)
    with registry._connect() as c:
        row = c.execute('select * from label_studio_accounts where owner_user_id=%s', (user['id'],)).fetchone()
    if not row:
        return {'connected': False, 'activated': False}
    return {'connected': True, **bridge(user['id'], 'account', {'email': user['email']}), 'url': os.environ.get('LABEL_STUDIO_PUBLIC_URL', 'http://172.25.2.135:8080') + '/projects/'}


@router.post('/account')
def create_account(request: Request):
    user = owner(request)
    result = bridge(user['id'], 'activate-link', {'email': user['email']})
    with registry._connect() as c:
        c.execute('insert into label_studio_accounts(owner_user_id,ls_user_id) values(%s,%s) on conflict(owner_user_id) do nothing', (user['id'], result['user_id']))
    return {'connected': True, **result}


async def upload(project_id: uuid.UUID, request: Request):
    user = await run_in_threadpool(owner, request)
    await run_in_threadpool(project_for, user['id'], project_id)
    if request.headers.get('content-type', '').split(';')[0] not in {'application/zip', 'application/octet-stream'}:
        raise HTTPException(415, 'Send the ZIP as the request body.')
    if int(request.headers.get('content-length', '0')) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'ZIP upload exceeds 2 GB.')
    root = contained_path(DATASET_DIR, '.label-studio-uploads')
    root.mkdir(parents=True, exist_ok=True)
    jobs.enough_disk(root, 0)
    descriptor, filename = tempfile.mkstemp(prefix='.upload-', suffix='.zip', dir=root)
    path = Path(filename)
    try:
        total = 0
        with os.fdopen(descriptor, 'wb') as output:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, 'ZIP upload exceeds 2 GB.')
                jobs.enough_disk(root, len(chunk))
                await run_in_threadpool(output.write, chunk)
        if not total:
            raise ValueError('The ZIP is empty.')
        return {'operation': await run_in_threadpool(jobs.create, user['id'], project_id, 'import', path)}
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@router.get('/operations/{operation_id}')
def get_operation(operation_id: uuid.UUID, request: Request):
    return {'operation': jobs.operation(owner(request)['id'], operation_id)}


@router.post('/operations/{operation_id}/retry', status_code=202)
def retry(operation_id: uuid.UUID, request: Request):
    return {'operation': jobs.retry(owner(request)['id'], operation_id)}


@router.post('/operations/{operation_id}/cancel')
def cancel(operation_id: uuid.UUID, request: Request):
    return {'operation': jobs.cancel(owner(request)['id'], operation_id)}
