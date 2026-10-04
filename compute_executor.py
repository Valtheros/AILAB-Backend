"""One fresh process per job, including inference, releases model memory on exit."""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from compute_repository import job_directory, registry
from security_utils import contained_path
from settings import RUNS_DIR
from worker.error_utils import concise_error


def run(job_id, token):
    directory = job_directory(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f'{token}.json'
    outcome = {'status': 'failed'}
    try:
        with registry._connect() as c:
            row = c.execute("select * from compute_jobs where id=%s and dispatch_token=%s and status='running'",
                            (job_id, token)).fetchone()
        if not row:
            return
        device = os.environ['COMPUTE_EXEC_DEVICE']
        import torch
        if device == 'cuda:0' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError('Assigned GPU is unavailable. CPU fallback is disabled.')
        payload = row['payload']
        if row['kind'] == 'train':
            from worker_app import run_training
            config = {**payload, 'compute_job_id': job_id,
                      'extra_args': {**payload.get('extra_args', {}), 'device': '0' if device == 'cuda:0' else 'cpu'}}
            run_training(config)
        else:
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
        outcome = {'status': 'completed', 'result_path': str(directory / 'result.json') if row['kind'] == 'predict' else None}
    except Exception as exc:
        import traceback
        (directory / 'error.log').write_text(traceback.format_exc()[-65536:], encoding='utf-8')
        text = concise_error(exc)
        outcome = {'status': 'failed', 'error_code': 'GPU_OOM' if 'out of memory' in str(exc).lower() else 'COMPUTE_FAILED',
                   'error_message': text}
    finally:
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(outcome), encoding='utf-8')
        temporary.replace(output)


if __name__ == '__main__':
    run(*sys.argv[1:])
