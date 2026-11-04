from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import compute_repository as compute
from api_errors import PublicError
from label_studio_api import owner

router = APIRouter(prefix='/api/compute')


class ExecutionSelection(BaseModel):
    mode: Literal['auto', 'gpu', 'cpu'] = 'auto'
    gpuUuid: str | None = Field(default=None, max_length=128)

    class Config:
        extra = 'forbid'


def admin(request):
    user = owner(request)
    try:
        compute.registry.assert_admin(user['id'])
    except PermissionError as exc:
        raise PublicError('ADMIN_REQUIRED', 'Administrator access is required.', status=403) from exc
    compute.require_enabled()
    return user


@router.get('/devices')
def devices(request: Request):
    owner(request)
    return compute.list_devices()


@router.get('/jobs/{job_id}')
def job(job_id: uuid.UUID, request: Request):
    return compute.get_job(owner(request)['id'], job_id)


@router.get('/jobs')
def latest(request: Request, runSlug: str = Query(max_length=128)):
    with compute.registry._connect() as c:
        row = c.execute('''select j.*,g.name as gpu_name from compute_jobs j join training_tasks t on j.run_id=t.id
          left join gpu_devices g on g.uuid=j.assigned_gpu_uuid
          where j.owner_user_id=%s and t.run_slug=%s and j.kind='predict' and j.cleaned_at is null
          order by j.created_at desc limit 1''', (owner(request)['id'], runSlug)).fetchone()
    return {'job': compute.public_job(row) if row else None}


@router.get('/jobs/{job_id}/input')
def input_image(job_id: uuid.UUID, request: Request):
    from PIL import Image
    row = compute.get_job(owner(request)['id'], job_id, public=False)
    path = compute.job_directory(job_id) / 'input'
    if row['kind'] != 'predict' or not path.is_file():
        raise PublicError('RESULT_EXPIRED', 'The input image is no longer available.', status=410)
    with Image.open(path) as image:
        mime = Image.MIME.get(image.format, 'application/octet-stream')
    return FileResponse(path, media_type=mime, headers={'Cache-Control': 'private, no-store'})


@router.get('/jobs/{job_id}/result')
def result(job_id: uuid.UUID, request: Request):
    return compute.read_result(owner(request)['id'], job_id)


@router.get('/jobs/{job_id}/overlay')
def overlay(job_id: uuid.UUID, request: Request):
    row = compute.get_job(owner(request)['id'], job_id, public=False)
    path = compute.job_directory(job_id) / 'overlay.png'
    if row['status'] != 'completed' or not path.is_file():
        raise PublicError('RESULT_EXPIRED', 'The overlay is not available.', status=410)
    return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'private, no-store'})


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id: uuid.UUID, request: Request):
    return compute.cancel_job(owner(request)['id'], job_id)


@router.get('/admin')
def administration(request: Request):
    admin(request)
    return compute.list_devices(admin=True)


class DeviceUpdate(BaseModel):
    draining: bool


@router.patch('/admin/devices/{gpu_uuid}')
def update_device(gpu_uuid: str, payload: DeviceUpdate, request: Request):
    admin(request)
    with compute.registry._connect() as c:
        if not c.execute('update gpu_devices set draining=%s where uuid=%s returning uuid', (payload.draining, gpu_uuid)).fetchone():
            raise PublicError('NOT_FOUND', 'GPU not found.', status=404)
    return {'updated': True}


class QuotaUpdate(BaseModel):
    maxGpuJobs: int = Field(ge=1, le=32)


@router.put('/admin/quotas/{user_id}')
def update_quota(user_id: str, payload: QuotaUpdate, request: Request):
    admin(request)
    with compute.registry._connect() as c:
        if not c.execute('select id from "user" where id=%s', (user_id,)).fetchone():
            raise PublicError('NOT_FOUND', 'User not found.', status=404)
        c.execute('insert into compute_user_quotas(owner_user_id,max_gpu_jobs) values(%s,%s) on conflict(owner_user_id) do update set max_gpu_jobs=excluded.max_gpu_jobs', (user_id, payload.maxGpuJobs))
    return {'updated': True}


class NodeUpdate(BaseModel):
    ramBudgetPercent: int = Field(ge=10, le=90)


@router.patch('/admin/nodes/{node_id}')
def update_node(node_id: str, payload: NodeUpdate, request: Request):
    admin(request)
    with compute.registry._connect() as c:
        if not c.execute('update compute_nodes set ram_budget_percent=%s where id=%s returning id', (payload.ramBudgetPercent, node_id)).fetchone():
            raise PublicError('NOT_FOUND', 'Node not found.', status=404)
    return {'updated': True}
