"""Stage a batch without loading its images into memory all at once."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import uuid
import zipfile
from pathlib import Path

from batch_inputs import MAX_BATCH_BYTES, MAX_BATCH_IMAGES, MAX_IMAGE_BYTES, extract_batch_zip, safe_image_name
from dataset_utils import IMAGE_EXTENSIONS
from api_errors import PublicError
from compute_repository import create_job, execution_selection, job_directory, registry, require_enabled
from resource_guard import estimate_training_memory
from security_utils import contained_path
from settings import RUNS_DIR


async def stage_batch(files, directory: Path) -> tuple[list[dict], str, int]:
    if not files or len(files) > MAX_BATCH_IMAGES:
        raise ValueError('Select between 1 and 100 images, or one image ZIP.')
    archive = len(files) == 1 and (files[0].filename or '').lower().endswith('.zip')
    if not archive and any((file.filename or '').lower().endswith('.zip') for file in files):
        raise ValueError('Upload a ZIP separately from other files.')
    input_dir = directory / 'inputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    entries, total = [], 0
    digest = hashlib.sha256()
    for index, file in enumerate(files):
        name = safe_image_name(file.filename or '')
        name_bytes = name.encode('utf-8', errors='replace')
        digest.update(len(name_bytes).to_bytes(4, 'big'))
        digest.update(name_bytes)
        file_hash = hashlib.sha256()
        destination = directory / 'input.zip' if archive else input_dir / str(index)
        size, too_large = 0, False
        valid_extension = archive or Path(name).suffix.lower() in IMAGE_EXTENSIONS
        with destination.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                total += len(chunk)
                if total > MAX_BATCH_BYTES:
                    raise ValueError('Batch exceeds the 1 GiB upload limit.')
                file_hash.update(chunk)
                if not archive and size > MAX_IMAGE_BYTES:
                    too_large = True
                if valid_extension and not too_large:
                    output.write(chunk)
        digest.update(size.to_bytes(8, 'big'))
        digest.update(file_hash.digest())
        if not archive:
            entry = {'name': name, 'bytes': size, 'sha256': file_hash.hexdigest()}
            if too_large or not valid_extension:
                destination.unlink(missing_ok=True)
                entry['error'] = 'Image exceeds the 10 MiB limit.' if too_large else 'Unsupported image format.'
            entries.append(entry)
    if archive:
        entries = await asyncio.to_thread(extract_batch_zip, directory / 'input.zip', directory)
        (directory / 'input.zip').unlink(missing_ok=True)
    (directory / 'input-manifest.json').write_text(json.dumps({'images': entries}, ensure_ascii=False), encoding='utf-8')
    return entries, digest.hexdigest(), total


async def submit_batch(owner_id: str, run: dict, files, selection: dict, threshold, key: str, *, principal=None):
    require_enabled()
    if run.get('status') != 'completed':
        raise PublicError('RUN_NOT_COMPLETED', 'Only completed runs can be tested.', status=409)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise PublicError('INVALID_THRESHOLD', 'Confidence threshold must be between 0 and 1.')
    if not key or len(key) > 128:
        raise PublicError('IDEMPOTENCY_REQUIRED', 'A valid request key is required.')
    selection = execution_selection(selection)
    run_dir = contained_path(RUNS_DIR, run['run_slug'])
    if not run_dir.is_dir():
        raise PublicError('NOT_FOUND', 'Run not found.', status=404)
    try:
        config = json.loads((run_dir / 'job_config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise PublicError('MODEL_CONFIGURATION_MISSING', 'The model configuration is unavailable.', status=409)
    job_id = uuid.uuid4()
    directory = job_directory(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    submission_attempted = False
    try:
        try:
            images, digest, total = await stage_batch(files, directory)
        except zipfile.BadZipFile as exc:
            raise PublicError('INVALID_BATCH', 'The ZIP is damaged or is not a valid ZIP file.', status=400) from exc
        except ValueError as exc:
            messages = {
                'Select between 1 and 100 images, or one image ZIP.',
                'Upload a ZIP separately from other files.',
                'Batch exceeds the 1 GiB upload limit.',
                'ZIP contains too many entries.',
                'ZIP contains an unsafe path.',
                'ZIP contains an unsafe absolute path',
                'ZIP symbolic links are not supported',
                'ZIP entry exceeds the maximum allowed size',
                'ZIP entry has an unsafe compression ratio',
                'Batch exceeds the image count or expanded size limit.',
                'ZIP contains no supported images.',
            }
            detail = str(exc) if str(exc) in messages else 'The batch contains an invalid or unsafe file.'
            status = 413 if detail == 'Batch exceeds the 1 GiB upload limit.' else 400
            raise PublicError('INVALID_BATCH', detail, status=status) from exc
        payload = {'project_name': run['run_slug'], 'model_type': run['model_type'], 'task_type': run['task_type'],
                   'threshold': threshold, 'input_sha256': digest, 'input_name': f'{len(images)} images', 'input_bytes': total}
        if principal:
            payload.update(checkpoint_path=principal['checkpoint_path'], checkpoint_sha256=principal['checkpoint_sha256'])
        submission_attempted = True
        row = await asyncio.to_thread(create_job, owner_id, run['id'], 'batch_predict', payload, selection,
                   idempotency_key=key, job_id=job_id,
                   model_endpoint_id=principal['endpoint_id'] if principal else None,
                   model_api_key_id=principal['key_id'] if principal else None,
                   estimated_vram_mb=estimate_training_memory(run['model_type'], config.get('extra_args', {}), 1))
        if str(row['id']) != str(job_id):
            shutil.rmtree(directory)
        return row
    except Exception:
        if not submission_attempted:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        try:
            with registry._connect() as connection:
                persisted = connection.execute('select id from compute_jobs where id=%s', (job_id,)).fetchone()
            if not persisted:
                shutil.rmtree(directory, ignore_errors=True)
        except Exception:
            pass
        raise
