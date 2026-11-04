"""Stage and queue one image for either the web application or model API."""
from __future__ import annotations

import asyncio
import hashlib
import math
import shutil
import uuid

from api_errors import PublicError
from batch_inputs import safe_image_name
import compute_repository as compute
from inference_service import InferenceError, _open_verified_image
from resource_guard import estimate_training_memory


async def submit_single(owner_id, run, config, payload, filename, selection, threshold, key, *, principal=None):
    compute.require_enabled()
    if run['status'] != 'completed':
        raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be tested.', status=409)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise PublicError('INVALID_THRESHOLD', 'Confidence threshold must be between 0 and 1.')
    if not key or len(key) > 128:
        raise PublicError('IDEMPOTENCY_REQUIRED', 'A valid request key is required.')
    try:
        _open_verified_image(payload).close()
    except InferenceError as exc:
        raise PublicError('INVALID_IMAGE', str(exc)) from exc
    selection = compute.execution_selection(selection)
    job_id = uuid.uuid4()
    directory = compute.job_directory(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    data = {'project_name': run['run_slug'], 'model_type': run['model_type'], 'task_type': run['task_type'],
            'threshold': threshold, 'input_sha256': hashlib.sha256(payload).hexdigest(),
            'input_name': safe_image_name(filename), 'input_bytes': len(payload)}
    if principal:
        data.update(checkpoint_path=principal['checkpoint_path'], checkpoint_sha256=principal['checkpoint_sha256'])
    try:
        (directory / 'input').write_bytes(payload)
        row = await asyncio.to_thread(compute.create_job, owner_id, run['id'], 'predict', data, selection,
            idempotency_key=key, job_id=job_id, estimated_vram_mb=estimate_training_memory(run['model_type'], config.get('extra_args', {}), 1),
            model_endpoint_id=principal['endpoint_id'] if principal else None, model_api_key_id=principal['key_id'] if principal else None)
        if str(row['id']) != str(job_id):
            shutil.rmtree(directory)
        return row
    except Exception:
        try:
            with compute.registry._connect() as connection:
                persisted = connection.execute('select id from compute_jobs where id=%s', (job_id,)).fetchone()
            if not persisted:
                shutil.rmtree(directory, ignore_errors=True)
        except Exception:
            pass
        raise
