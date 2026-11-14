"""Opt-in CPU smoke through the public API and an isolated compute worker."""

import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

from tests.isolation import require_test_database, require_test_redis, require_test_storage


def _environment():
    from resource_repository import resource_repository
    from settings import REDIS_URL, RUNS_DIR

    require_test_database(resource_repository.database_url)
    require_test_redis(REDIS_URL)
    require_test_storage('RUNS_DIR', 'DATASET_DIR', 'STORAGE_DIR', 'APP_BASE_DIR', 'TMPDIR')
    return resource_repository, RUNS_DIR


def _metadata_path():
    token = uuid.UUID(os.environ['AILAB_SMOKE_ID'])
    return Path(os.environ['AILAB_TEST_ROOT']) / f'real-api-smoke-{token}.json'


def seed():
    import torch
    from inference_service import _build_classifier
    from scripts.apply_schema import apply

    registry, runs = _environment()
    slug = f'api_smoke_{uuid.uuid4().hex}'
    owner = f'api-smoke-{uuid.uuid4().hex}'
    directory = runs / slug
    directory.mkdir(parents=True, exist_ok=False)
    config = {'model_type': 'resnet', 'model_name': 'resnet18',
              'extra_args': {'architecture': 'resnet18', 'image_size': 32, 'dropout': 0.2}}
    model = _build_classifier('resnet18', 2, 0.2)
    torch.save({'model': model.state_dict(), 'config': config, 'classes': ['one', 'two']}, directory / 'best.pt')
    (directory / 'job_config.json').write_text(json.dumps(config))
    with registry._connect() as connection:
        connection.execute('''create table if not exists "user"(id text primary key,email text unique,name text,
            "emailVerified" boolean default true,banned boolean default false,role text default 'user')''')
        apply(connection)
        connection.execute('insert into "user"(id,email) values(%s,%s)', (owner, f'{owner}@example.invalid'))
        connection.execute('''insert into training_tasks(owner_user_id,created_by,run_slug,task_type,model_type,status)
            values(%s,%s,%s,'image_classification','resnet','completed')''', (owner, owner, slug))
    with _metadata_path().open('x') as metadata:
        json.dump({'owner': owner, 'runSlug': slug}, metadata)
    print('Prepared an isolated, randomly initialized ResNet checkpoint.')


def submit():
    from unittest.mock import patch
    from PIL import Image
    from fastapi.testclient import TestClient
    import model_api_service

    registry, _ = _environment()
    metadata = json.loads(_metadata_path().read_text())
    from main import app

    image = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(image, format='PNG')
    session = {'x-internal-token': os.environ['BACKEND_INTERNAL_TOKEN'], 'x-user-id': metadata['owner']}
    with patch.object(model_api_service, 'enabled', return_value=True), TestClient(app) as client:
        endpoint = client.post('/api/model-endpoints', headers=session, json={'runSlug': metadata['runSlug']})
        if endpoint.status_code != 201:
            raise RuntimeError(f'Endpoint setup failed: {endpoint.status_code} {endpoint.text}')
        endpoint_id = endpoint.json()['id']
        key = client.post(f'/api/model-endpoints/{endpoint_id}/keys', headers=session, json={'label': 'smoke'})
        if key.status_code != 201:
            raise RuntimeError(f'Key setup failed: {key.status_code} {key.text}')
        headers = {'authorization': f"Bearer {key.json()['secret']}", 'idempotency-key': 'smoke-single'}

        def wait_for_job(response):
            if response.status_code != 202:
                raise RuntimeError(f'Submission failed: {response.status_code} {response.text}')
            job_id = response.json()['id']
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                status = client.get(f'/api/v1/jobs/{job_id}', headers=headers)
                if status.status_code != 200:
                    raise RuntimeError(f'Job lookup failed: {status.status_code} {status.text}')
                job = status.json()
                if job['status'] == 'completed':
                    result = client.get(f'/api/v1/jobs/{job_id}/result', headers=headers)
                    if result.status_code != 200:
                        raise RuntimeError(f'Result lookup failed: {result.status_code} {result.text}')
                    return result.json()
                if job['status'] in {'failed', 'cancelled'}:
                    raise RuntimeError(f"Job failed: {job.get('errorCode')} {job.get('error')}")
                time.sleep(2)
            raise TimeoutError(f'Job did not complete: {job}')

        single = client.post(f'/api/v1/models/{endpoint_id}/predict', headers=headers,
                             data={'execution': '{"mode":"cpu"}'},
                             files={'file': ('image.png', image.getvalue(), 'image/png')})
        result = wait_for_job(single)
        if result['model']['device'] != 'cpu' or not result.get('predictions'):
            raise RuntimeError('Single prediction did not use the CPU worker')

        headers['idempotency-key'] = 'smoke-batch'
        batch = client.post(f'/api/v1/models/{endpoint_id}/predict-batch', headers=headers,
                            data={'execution': '{"mode":"cpu"}'}, files=[
                                ('files', ('image.png', image.getvalue(), 'image/png')),
                                ('files', ('bad.png', b'not an image', 'image/png'))])
        result = wait_for_job(batch)
        if (result['succeeded'], result['failed']) != (1, 1):
            raise RuntimeError('Batch did not preserve the per-image failure')
        print('Public API -> PostgreSQL -> Redis -> CPU worker -> results: single and partial batch passed.')
        print('Quota gate was mocked; this is not a production quota test.')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'seed', 'submit'}:
        raise SystemExit('Use seed or submit in an isolated test stack')
    if os.getenv('COMPUTE_TESTS') != '1':
        raise SystemExit('COMPUTE_TESTS=1 is required')
    {'seed': seed, 'submit': submit}[sys.argv[1]]()
