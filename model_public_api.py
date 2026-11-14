from __future__ import annotations

import hashlib
import json
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api_errors import PublicError
from batch_upload import submit_batch
from compute_api import batch_result_file
import compute_repository as compute
from inference_service import MAX_INFERENCE_IMAGE_BYTES
from prediction_submission import submit_single
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import RUNS_DIR

router = APIRouter(prefix='/api/v1')


def _principal(request):
    return request.state.model_api_principal


def _request_key(request, principal):
    value = request.headers.get('idempotency-key', '')
    if not value or len(value) > 128 or any(ord(character) < 32 for character in value):
        raise PublicError('IDEMPOTENCY_REQUIRED', 'A valid request key is required.')
    digest = hashlib.sha256(value.encode('utf-8')).hexdigest()
    return f"api:{principal['key_id']}:{digest}"


def _selection(value):
    try:
        selection = json.loads(value) if value else {'mode': 'auto'}
        if not isinstance(selection, dict):
            raise ValueError('Invalid device selection')
        return compute.execution_selection(selection)
    except (TypeError, ValueError) as exc:
        raise PublicError('INVALID_EXECUTION', 'Choose Auto GPU, a GPU, or CPU.') from exc


def _model(request, endpoint_id):
    principal = _principal(request)
    if str(principal['endpoint_id']) != str(endpoint_id):
        raise PublicError('NOT_FOUND', 'Model not found.', status=404)
    if not principal['enabled'] or principal['run_status'] != 'completed':
        raise PublicError('MODEL_API_UNAVAILABLE', 'This model endpoint is not accepting new jobs.', status=409)
    run = registry.get_run(principal['owner_user_id'], principal['run_slug'])
    if not run or str(run['id']) != str(principal['run_id']):
        raise PublicError('NOT_FOUND', 'Model not found.', status=404)
    return principal, run


def _job(request, job_id):
    principal = _principal(request)
    row = compute.get_job(principal['owner_user_id'], job_id, public=False)
    if str(row.get('model_api_key_id')) != str(principal['key_id']) or str(row.get('model_endpoint_id')) != str(principal['endpoint_id']):
        raise PublicError('NOT_FOUND', 'Job not found.', status=404)
    compute.require_unexpired_result(row)
    return principal, row


def _accepted(row):
    result = compute.public_job(row)
    result.update(statusUrl=f"/api/v1/jobs/{row['id']}", resultUrl=f"/api/v1/jobs/{row['id']}/result",
                  cancelUrl=f"/api/v1/jobs/{row['id']}/cancel")
    return JSONResponse(json.loads(json.dumps(result, default=str)), status_code=202,
                        headers={'Cache-Control': 'no-store', 'Location': result['statusUrl']})


@router.post('/models/{endpoint_id}/predict')
async def predict(endpoint_id: uuid.UUID, request: Request, file: UploadFile = File(...),
                  threshold: float | None = Form(None), execution: str | None = Form(None)):
    principal, run = _model(request, endpoint_id)
    key = _request_key(request, principal)
    payload = await file.read(MAX_INFERENCE_IMAGE_BYTES + 1)
    if len(payload) > MAX_INFERENCE_IMAGE_BYTES:
        raise PublicError('IMAGE_TOO_LARGE', 'The image exceeds the 10 MiB limit.', status=413)
    config_path = contained_path(RUNS_DIR, run['run_slug'], 'job_config.json')
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise PublicError('MODEL_CONFIGURATION_MISSING', 'The model configuration is unavailable.', status=409) from exc
    row = await submit_single(principal['owner_user_id'], run, config, payload, file.filename or 'image',
                              _selection(execution), threshold, key, principal=principal)
    return _accepted(row)


@router.post('/models/{endpoint_id}/predict-batch')
async def predict_batch(endpoint_id: uuid.UUID, request: Request, files: list[UploadFile] = File(...),
                        threshold: float | None = Form(None), execution: str | None = Form(None)):
    principal, run = _model(request, endpoint_id)
    row = await submit_batch(principal['owner_user_id'], run, files, _selection(execution), threshold,
                             _request_key(request, principal), principal=principal)
    return _accepted(row)


@router.get('/jobs/{job_id}')
def job(job_id: uuid.UUID, request: Request):
    _, row = _job(request, job_id)
    result = compute.public_job(row)
    result.update(resultUrl=f'/api/v1/jobs/{job_id}/result', cancelUrl=f'/api/v1/jobs/{job_id}/cancel')
    return result


@router.get('/jobs/{job_id}/result')
def result(job_id: uuid.UUID, request: Request):
    principal, row = _job(request, job_id)
    if row['kind'] == 'batch_predict':
        path = batch_result_file(job_id, principal['owner_user_id'], 'result.json')
        result = json.loads(path.read_text(encoding='utf-8'))
    else:
        result = compute.read_result(principal['owner_user_id'], job_id)
        if result.get('segmentation', {}).get('overlay'):
            result['segmentation']['overlay'] = f'/api/v1/jobs/{job_id}/files/overlay.png'
    return result


@router.get('/jobs/{job_id}/files/{file_path:path}')
def file(job_id: uuid.UUID, file_path: str, request: Request):
    principal, row = _job(request, job_id)
    if row['kind'] == 'batch_predict':
        path = batch_result_file(job_id, principal['owner_user_id'], file_path)
    else:
        if file_path not in {'input', 'overlay.png'}:
            raise PublicError('NOT_FOUND', 'File not found.', status=404)
        path = contained_path(compute.job_directory(job_id), file_path)
        if not path.is_file() or (file_path == 'overlay.png' and row['status'] != 'completed'):
            raise PublicError('RESULT_EXPIRED', 'The file is unavailable.', status=410)
    return FileResponse(path, filename=path.name, headers={'Cache-Control': 'private, no-store'})


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id: uuid.UUID, request: Request):
    principal, _ = _job(request, job_id)
    return compute.cancel_job(principal['owner_user_id'], job_id)
