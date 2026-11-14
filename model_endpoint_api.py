from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api_errors import PublicError
import compute_repository as compute
from label_studio_api import owner
import model_api_service as service
from resource_repository import resource_repository as registry

router = APIRouter(prefix='/api/model-endpoints')


class EndpointRequest(BaseModel):
    runSlug: str = Field(min_length=1, max_length=128)


class EndpointUpdate(BaseModel):
    enabled: bool


class KeyRequest(BaseModel):
    label: str = Field(default='', max_length=100)


def _available():
    if not service.enabled():
        raise PublicError('MODEL_API_DISABLED', 'Model API is not available until storage protection and deployment are ready.', status=503)
    compute.require_enabled()


def _public(row):
    return {'id': str(row['id']), 'runId': str(row['run_id']), 'enabled': row['enabled'],
            'checkpointSha256': row['checkpoint_sha256'], 'createdAt': row['created_at'], 'updatedAt': row['updated_at']}


def _endpoint(connection, subject, endpoint_id, lock=False):
    row = connection.execute('''select e.*,t.run_slug,t.model_type,t.status as run_status from model_endpoints e
        join training_tasks t on t.id=e.run_id where e.id=%s and e.owner_user_id=%s and t.owner_user_id=%s'''
        + (' for update of e' if lock else ''), (endpoint_id, subject, subject)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Model endpoint not found.', status=404)
    return row


@router.get('')
def list_endpoints(request: Request, runSlug: str):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        rows = connection.execute('''select e.* from model_endpoints e join training_tasks t on t.id=e.run_id
            where e.owner_user_id=%s and t.owner_user_id=%s and t.run_slug=%s order by e.created_at desc''',
            (subject, subject, runSlug)).fetchall()
    return {'endpoints': [_public(row) for row in rows]}


@router.post('', status_code=201)
def create_endpoint(payload: EndpointRequest, request: Request):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
        run = connection.execute('''select id,run_slug,model_type,status from training_tasks where owner_user_id=%s
            and run_slug=%s for update''', (subject, payload.runSlug)).fetchone()
        if not run:
            raise PublicError('NOT_FOUND', 'Run not found.', status=404)
        if run['status'] != 'completed':
            raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be exposed.', status=409)
        previous = connection.execute('select * from model_endpoints where run_id=%s', (run['id'],)).fetchone()
        if previous:
            raise PublicError('ENDPOINT_EXISTS', 'This run already has a model endpoint.', status=409)
        try:
            path, digest = service.checkpoint(run['run_slug'], run['model_type'])
        except (OSError, ValueError) as exc:
            raise PublicError('CHECKPOINT_UNAVAILABLE', 'The saved checkpoint is unavailable.', status=409) from exc
        row = connection.execute('''insert into model_endpoints(owner_user_id,run_id,checkpoint_path,checkpoint_sha256)
            values(%s,%s,%s,%s) returning *''', (subject, run['id'], path, digest)).fetchone()
    return _public(row)


@router.patch('/{endpoint_id}')
def update_endpoint(endpoint_id: uuid.UUID, payload: EndpointUpdate, request: Request):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
        row = _endpoint(connection, subject, endpoint_id, lock=True)
        if payload.enabled:
            if row['run_status'] != 'completed':
                raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be exposed.', status=409)
            try:
                service.verify_checkpoint(row['run_slug'], row['model_type'], row['checkpoint_path'], row['checkpoint_sha256'])
            except OSError as exc:
                raise PublicError('CHECKPOINT_UNAVAILABLE', 'The saved checkpoint is unavailable.', status=409) from exc
        row = connection.execute('update model_endpoints set enabled=%s,updated_at=now() where id=%s returning *',
                                 (payload.enabled, endpoint_id)).fetchone()
    return _public(row)


@router.delete('/{endpoint_id}')
def delete_endpoint(endpoint_id: uuid.UUID, request: Request):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
        row = _endpoint(connection, subject, endpoint_id, lock=True)
        if row['enabled'] or connection.execute('''select 1 from compute_jobs where model_endpoint_id=%s
            and status=any(%s) limit 1''', (endpoint_id, ['queued', *compute.ACTIVE])).fetchone():
            raise PublicError('MODEL_API_IN_USE', 'Disable the endpoint and wait for its jobs to finish.', status=409)
        connection.execute('delete from model_endpoints where id=%s', (endpoint_id,))
    return {'deleted': True}


@router.get('/{endpoint_id}/keys')
def list_keys(endpoint_id: uuid.UUID, request: Request):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        _endpoint(connection, subject, endpoint_id)
        rows = connection.execute('''select id,label,created_at,expires_at,revoked_at from model_api_keys
            where endpoint_id=%s order by created_at desc''', (endpoint_id,)).fetchall()
    return {'keys': [dict(row) for row in rows]}


@router.post('/{endpoint_id}/keys', status_code=201)
def create_key(endpoint_id: uuid.UUID, payload: KeyRequest, request: Request):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
        _endpoint(connection, subject, endpoint_id, lock=True)
        row, secret = service.issue_key(connection, endpoint_id, payload.label.strip())
    return JSONResponse(jsonable_encoder({'key': row, 'secret': secret}), status_code=201,
                        headers={'Cache-Control': 'private, no-store', 'Pragma': 'no-cache'})


@router.post('/{endpoint_id}/keys/{key_id}/revoke')
def revoke_key(endpoint_id: uuid.UUID, key_id: uuid.UUID, request: Request):
    _available()
    subject = owner(request)['id']
    with registry._connect() as connection:
        connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
        _endpoint(connection, subject, endpoint_id, lock=True)
        row = connection.execute('''update model_api_keys set revoked_at=coalesce(revoked_at,now())
            where id=%s and endpoint_id=%s returning id''', (key_id, endpoint_id)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Model API key not found.', status=404)
    return {'revoked': True}
