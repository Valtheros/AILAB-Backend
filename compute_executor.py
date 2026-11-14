"""One fresh process per job, including inference, releases model memory on exit."""
from __future__ import annotations

import base64
import errno
import json
import os
import sys
from pathlib import Path

from compute_repository import job_directory, registry
from dataset_storage import registered_storage_path
from dataset_versioning import verify_source
from security_utils import contained_path
from settings import DATASET_DIR, RUNS_DIR
from worker.error_utils import concise_error


def report_outcome(output, outcome):
    data = json.dumps(outcome, ensure_ascii=False).encode('utf-8')
    if len(data) > 4096:
        data = json.dumps({'status': 'failed', 'error_code': 'COMPUTE_FAILED',
                           'error_message': 'The worker result summary exceeded its size limit.'}).encode('utf-8')
    report_fd = os.environ.get('COMPUTE_OUTCOME_FD')
    if report_fd is not None:
        try:
            os.write(int(report_fd), data)
        finally:
            os.close(int(report_fd))
    try:
        temporary = output.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(output)
    except OSError:
        if report_fd is None:
            raise


def run(job_id, token):
    directory = job_directory(job_id)
    with registry._connect() as c:
        row = c.execute("select * from compute_jobs where id=%s and dispatch_token=%s and status='running'",
                        (job_id, token)).fetchone()
    if not row:
        return
    output = directory / f'{token}.json'
    outcome = {'status': 'failed'}
    try:
        directory.mkdir(parents=True, exist_ok=True)
        device = os.environ['COMPUTE_EXEC_DEVICE']
        import torch
        if device == 'cuda:0' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError('Assigned GPU is unavailable. CPU fallback is disabled.')
        payload = row['payload']
        if row['kind'] in {'predict', 'batch_predict'} and payload.get('checkpoint_sha256'):
            from model_api_service import verify_checkpoint
            verify_checkpoint(payload['project_name'], payload['model_type'], payload['checkpoint_path'], payload['checkpoint_sha256'])
        if row['kind'] == 'train':
            from worker_app import run_training
            config = {**payload, 'compute_job_id': job_id,
                      'extra_args': {**payload.get('extra_args', {}), 'device': '0' if device == 'cuda:0' else 'cpu'}}
            expected = payload.get('dataset_content_sha256')
            source = registered_storage_path(DATASET_DIR, payload['source_dataset_path']) if expected else None
            dataset_verified = verify_source(source, expected) if source else False
            run_training(config)
            if source:
                verify_source(source, expected)
        elif row['kind'] == 'predict':
            run_dir = contained_path(RUNS_DIR, payload['project_name'])
            image = (directory / 'input').read_bytes()
            model, task = payload['model_type'], payload['task_type']
            if task == 'image_classification':
                from inference_service import predict_image
                result = predict_image(run_dir, image, device=device)
            elif task == 'object_detection':
                from detection_inference import predict_detection
                result = predict_detection(run_dir, image, model, payload.get('threshold'), device=device)
            else:
                from segmentation_inference import predict_segmentation
                result = predict_segmentation(run_dir, image, model, payload.get('threshold'), device=device)
            segmentation = result.get('segmentation', {})
            if segmentation.get('overlay', '').startswith('data:image/png;base64,'):
                (directory / 'overlay.png').write_bytes(base64.b64decode(segmentation['overlay'].split(',', 1)[1], validate=True))
                segmentation['overlay'] = f'/api/backend/api/compute/jobs/{job_id}/overlay'
            result.update(status='success', taskType=task, runSlug=payload['project_name'])
            (directory / 'result.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
        elif row['kind'] == 'evaluate':
            from evaluation_executor import evaluate_job
            result_path, result_state = evaluate_job(row, device)
            outcome = {'status': 'completed', 'result_path': result_path, 'result_state': result_state}
        elif row['kind'] == 'batch_predict':
            from batch_prediction import execute_batch
            execute_batch(row, device)
        else:
            raise ValueError('Unsupported compute job type.')
        if row['kind'] in {'predict', 'batch_predict'} and payload.get('checkpoint_sha256'):
            verify_checkpoint(payload['project_name'], payload['model_type'], payload['checkpoint_path'], payload['checkpoint_sha256'])
        if row['kind'] != 'evaluate':
            outcome = {'status': 'completed', 'result_path': str(directory / 'result.json') if row['kind'] in {'predict', 'batch_predict'} else None}
            if row['kind'] == 'train':
                outcome['dataset_verified'] = dataset_verified
    except Exception as exc:
        import traceback
        try:
            (directory / 'error.log').write_text(traceback.format_exc()[-65536:], encoding='utf-8')
        except OSError:
            pass
        text = concise_error(exc)
        storage_full = isinstance(exc, OSError) and exc.errno in {errno.EDQUOT, errno.ENOSPC}
        code = ('STORAGE_QUOTA_EXCEEDED' if storage_full else 'GPU_OOM' if 'out of memory' in str(exc).lower()
                else 'CHECKPOINT_CHANGED' if getattr(exc, 'code', None) == 'CHECKPOINT_CHANGED' else 'COMPUTE_FAILED')
        if storage_full:
            text = 'The job ran out of storage allowance. Free space or ask an administrator to increase your quota, then retry.'
        outcome = {'status': 'failed', 'error_code': code,
                   'error_message': text}
    finally:
        report_outcome(output, outcome)


if __name__ == '__main__':
    run(*sys.argv[1:])
