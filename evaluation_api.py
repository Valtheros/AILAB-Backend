from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api_errors import PublicError
from compute_api import ExecutionSelection
from dataset_storage import registered_storage_path
from dataset_utils import compatible_models_for_metadata
from evaluation_repository import cancel, create_manual, public_evaluation
from label_studio_api import owner
from image_viewer import viewer_image
from model_catalog import get_catalog
from resource_repository import resource_repository as registry
from security_utils import ReversibleDirectoryRemoval, contained_path, validate_slug
from settings import DATASET_DIR, RUNS_DIR

router = APIRouter(prefix='/api')


class EvaluationRequest(BaseModel):
    datasetVersionId: uuid.UUID
    execution: ExecutionSelection | None = None


def _evaluation(owner_id, evaluation_id):
    with registry._connect() as connection:
        row = connection.execute('''select e.*,t.run_slug,d.slug as dataset_slug,d.storage_path,d.status as dataset_status,
            d.version,c.name as collection_name from evaluations e join training_tasks t on t.id=e.run_id
            join datasets d on d.id=e.dataset_version_id left join dataset_collections c on c.id=d.collection_id
            where e.id=%s and e.owner_user_id=%s''', (evaluation_id, owner_id)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Evaluation not found.', status=404)
    return dict(row)


def _public(row):
    result = public_evaluation(row)
    result.update(runSlug=row['run_slug'], datasetName=row['collection_name'] or row['dataset_slug'],
                  datasetVersion=row['version'], datasetStatus=row['dataset_status'])
    return result


@router.post('/runs/{run_slug}/evaluations', status_code=202)
def create_evaluation(run_slug: str, payload: EvaluationRequest, request: Request):
    subject = owner(request)['id']
    try:
        validate_slug(run_slug, 'run name')
    except ValueError as exc:
        raise PublicError('NOT_FOUND', 'Run not found.', status=404) from exc
    run = registry.get_run(subject, run_slug)
    with registry._connect() as connection:
        dataset = connection.execute("select metadata from datasets where id=%s and owner_user_id=%s and status='active'",
                                     (payload.datasetVersionId, subject)).fetchone()
    if not run or not dataset:
        raise PublicError('NOT_FOUND', 'Run or dataset version not found.', status=404)
    compatible = next((model for model in compatible_models_for_metadata(dataset['metadata'], get_catalog())
                       if model['id'] == run['model_type']), None)
    if not compatible or not compatible['ready']:
        raise PublicError('DATASET_INCOMPATIBLE', 'This dataset is not compatible with the trained model.', status=400)
    selection = payload.execution.model_dump() if payload.execution else {'mode': 'auto', 'gpuUuid': None}
    return create_manual(subject, run['id'], payload.datasetVersionId, selection, request.headers.get('idempotency-key', ''))


@router.get('/runs/{run_slug}/evaluations')
def list_evaluations(run_slug: str, request: Request):
    subject = owner(request)['id']
    run = registry.get_run(subject, run_slug)
    if not run:
        raise PublicError('NOT_FOUND', 'Run not found.', status=404)
    with registry._connect() as connection:
        rows = connection.execute('''select e.*,t.run_slug,d.slug as dataset_slug,d.storage_path,d.status as dataset_status,
            d.version,c.name as collection_name from evaluations e join training_tasks t on t.id=e.run_id
            join datasets d on d.id=e.dataset_version_id left join dataset_collections c on c.id=d.collection_id
            where e.owner_user_id=%s and e.run_id=%s order by e.created_at desc''', (subject, run['id'])).fetchall()
    return {'evaluations': [_public(row) for row in rows]}


@router.get('/evaluations/{evaluation_id}')
def get_evaluation(evaluation_id: uuid.UUID, request: Request):
    row = _evaluation(owner(request)['id'], evaluation_id)
    result = _public(row)
    if row['result_path'] and row['status'] in {'completed', 'partial', 'no_test'}:
        path = contained_path(RUNS_DIR, Path(row['result_path']))
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            raise PublicError('RESULT_UNAVAILABLE', 'The evaluation result is unavailable.', status=410)
        result['report'] = json.loads(path.read_text(encoding='utf-8'))
    return result


@router.post('/evaluations/{evaluation_id}/cancel')
def cancel_evaluation(evaluation_id: uuid.UUID, request: Request):
    return cancel(owner(request)['id'], evaluation_id)


@router.delete('/evaluations/{evaluation_id}')
def delete_evaluation(evaluation_id: uuid.UUID, request: Request):
    subject = owner(request)['id']
    removal = None
    try:
        with registry._connect() as connection:
            connection.execute('select pg_advisory_xact_lock(%s)', (registry._lock_key('user-resources:' + subject),))
            row = connection.execute('''select e.*,t.run_slug from evaluations e join training_tasks t on t.id=e.run_id
                where e.id=%s and e.owner_user_id=%s for update of e''', (evaluation_id, subject)).fetchone()
            if not row:
                raise PublicError('NOT_FOUND', 'Evaluation not found.', status=404)
            if row['status'] in {'awaiting_queue', 'queued', 'running', 'stopping'}:
                raise PublicError('EVALUATION_IN_USE', 'Cancel the evaluation before deleting its result.', status=409)
            target = contained_path(RUNS_DIR, row['run_slug'], 'evaluations', str(evaluation_id))
            if target.exists():
                removal = ReversibleDirectoryRemoval(target).apply()
            connection.execute('delete from evaluations where id=%s and owner_user_id=%s', (evaluation_id, subject))
        if removal:
            removal.commit()
    except Exception:
        if removal:
            removal.rollback()
        raise
    return {'deleted': True}


@router.get('/evaluations/{evaluation_id}/images')
def evaluation_images(evaluation_id: uuid.UUID, request: Request,
                      outcome: Literal['all', 'correct', 'incorrect', 'error', 'fp', 'fn'] = 'all',
                      sort: Literal['ordinal', 'score'] = 'ordinal', offset: int = Query(default=0, ge=0),
                      limit: int = Query(default=48, ge=1, le=100)):
    row = _evaluation(owner(request)['id'], evaluation_id)
    if row['status'] not in {'completed', 'partial'}:
        raise PublicError('RESULT_NOT_READY', 'Evaluation images are not ready.', status=409)
    condition = (' and fp_count>0' if outcome == 'fp' else ' and fn_count>0' if outcome == 'fn'
                 else '' if outcome == 'all' else ' and outcome=%s')
    args = [evaluation_id] + ([] if outcome in {'all', 'fp', 'fn'} else [outcome])
    order = 'score asc nulls last,ordinal' if sort == 'score' else 'ordinal'
    with registry._connect() as connection:
        total = connection.execute(f'select count(*) as count from evaluation_images where evaluation_id=%s{condition}', args).fetchone()['count']
        items = connection.execute(f'''select ordinal,filename,outcome,score,fp_count,fn_count from evaluation_images where evaluation_id=%s{condition}
                                     order by {order} limit %s offset %s''', [*args, limit, offset]).fetchall()
    return {'images': [dict(item) for item in items], 'total': total, 'offset': offset, 'limit': limit}


@router.get('/evaluations/{evaluation_id}/images/{ordinal}')
def evaluation_image(evaluation_id: uuid.UUID, ordinal: int, request: Request):
    row = _evaluation(owner(request)['id'], evaluation_id)
    if row['status'] not in {'completed', 'partial'} or not row['result_path']:
        raise PublicError('RESULT_NOT_READY', 'Evaluation images are not ready.', status=409)
    with registry._connect() as connection:
        image = connection.execute('select * from evaluation_images where evaluation_id=%s and ordinal=%s', (evaluation_id, ordinal)).fetchone()
    if not image:
        raise PublicError('NOT_FOUND', 'Image result not found.', status=404)
    if image['record_length'] < 0 or image['record_length'] > 10 * 1024 * 1024:
        raise PublicError('RESULT_UNAVAILABLE', 'Image result is unavailable.', status=410)
    path = contained_path(RUNS_DIR, Path(row['result_path'])).parent
    path = contained_path(path, 'images.jsonl')
    try:
        with path.open('rb') as stream:
            stream.seek(image['record_offset'])
            return json.loads(stream.read(image['record_length']))
    except (OSError, ValueError) as exc:
        raise PublicError('RESULT_UNAVAILABLE', 'Image result is unavailable.', status=410) from exc


@router.get('/evaluations/{evaluation_id}/images/{ordinal}/masks/{kind}')
def evaluation_image_mask(evaluation_id: uuid.UUID, ordinal: int, kind: Literal['actual', 'predicted'], request: Request):
    row = _evaluation(owner(request)['id'], evaluation_id)
    if row['status'] not in {'completed', 'partial'} or not row['result_path'] or ordinal < 0:
        raise PublicError('RESULT_NOT_READY', 'Evaluation images are not ready.', status=409)
    with registry._connect() as connection:
        image = connection.execute('select 1 from evaluation_images where evaluation_id=%s and ordinal=%s',
                                   (evaluation_id, ordinal)).fetchone()
    if not image:
        raise PublicError('NOT_FOUND', 'Image result not found.', status=404)
    base = contained_path(RUNS_DIR, Path(row['result_path'])).parent
    path = contained_path(base, 'masks', f'{ordinal}-{kind}.png')
    if not path.is_file():
        raise PublicError('RESULT_UNAVAILABLE', 'Image mask is unavailable.', status=410)
    return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'private, no-store'})


@router.get('/evaluations/{evaluation_id}/images/{ordinal}/source')
def evaluation_image_source(evaluation_id: uuid.UUID, ordinal: int, request: Request):
    row = _evaluation(owner(request)['id'], evaluation_id)
    if row['dataset_status'] != 'active':
        raise PublicError('DATASET_REMOVED', 'The original dataset has been removed.', status=410)
    with registry._connect() as connection:
        image = connection.execute('select source_path from evaluation_images where evaluation_id=%s and ordinal=%s', (evaluation_id, ordinal)).fetchone()
    if not image:
        raise PublicError('NOT_FOUND', 'Image not found.', status=404)
    root = registered_storage_path(DATASET_DIR, row['storage_path'])
    path = contained_path(root, image['source_path'])
    if not path.is_file():
        raise PublicError('DATASET_REMOVED', 'The original image is no longer available.', status=410)
    return viewer_image(path)
