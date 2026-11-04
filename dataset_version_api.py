from __future__ import annotations

import re
import uuid
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from api_errors import PublicError
from dataset_index_jobs import public, request_index
from dataset_storage import registered_storage_path
from label_studio_api import owner
from image_viewer import viewer_image
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import DATASET_DIR

router = APIRouter(prefix='/api/dataset-versions')


def _dataset(subject, dataset_id):
    with registry._connect() as connection:
        row = connection.execute('''select d.*,c.name as collection_name,i.status as index_status,
            i.active_generation,i.total as index_total,i.processed as index_processed,i.skipped as index_skipped,
            i.stats as index_stats,i.error_message as index_error from datasets d
            left join dataset_collections c on c.id=d.collection_id and c.owner_user_id=d.owner_user_id
            left join dataset_indexes i on i.dataset_version_id=d.id and i.owner_user_id=d.owner_user_id
            where d.id=%s and d.owner_user_id=%s and d.status='active' ''', (dataset_id, subject)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Dataset version not found.', status=404)
    return dict(row)


def _ready(subject, dataset_id):
    dataset = _dataset(subject, dataset_id)
    if not dataset['active_generation']:
        raise PublicError('INDEX_NOT_READY', 'Dataset details are still being prepared.', status=409)
    return dataset


@router.get('/{dataset_id}')
def detail(dataset_id: uuid.UUID, request: Request):
    row = _dataset(owner(request)['id'], dataset_id)
    metadata = row['metadata'] or {}
    return {'id': str(row['id']), 'name': row['collection_name'] or row['name'], 'version': row['version'],
            'contentSha256': row['content_sha256'],
            'metadata': {key: metadata.get(key) for key in ('formats', 'tasks', 'classes', 'label_schema')},
            'index': {'status': row['index_status'] or 'not_started', 'total': row['index_total'] or 0,
                      'processed': row['index_processed'] or 0, 'skipped': row['index_skipped'] or 0,
                      'stats': row['index_stats'] or {}, 'error': row['index_error'],
                      'ready': row['active_generation'] is not None}}


@router.post('/{dataset_id}/index', status_code=202)
def index(dataset_id: uuid.UUID, request: Request):
    return {'index': request_index(owner(request)['id'], dataset_id)}


@router.get('/{dataset_id}/images')
def images(dataset_id: uuid.UUID, request: Request, split: Literal['all', 'train', 'val', 'test'] = 'all',
           status: Literal['all', 'empty'] = 'all', className: str = Query(default='', max_length=512),
           search: str = Query(default='', max_length=128), sort: Literal['name', 'split'] = 'name',
           offset: int = Query(default=0, ge=0), limit: int = Query(default=48, ge=1, le=48)):
    row = _ready(owner(request)['id'], dataset_id)
    conditions = ['dataset_version_id=%s', 'generation=%s']
    args = [dataset_id, row['active_generation']]
    if split != 'all':
        conditions.append('split=%s'); args.append(split)
    if status == 'empty':
        conditions.append('not has_objects')
    if className:
        conditions.append('%s=any(class_names)'); args.append(className)
    if search:
        escaped = search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        conditions.append("filename ilike %s escape '\\'"); args.append(f'%{escaped}%')
    where = ' and '.join(conditions)
    order = 'filename,ordinal' if sort == 'name' else 'split,filename,ordinal'
    with registry._connect() as connection:
        total = connection.execute(f'select count(*) as count from dataset_index_images where {where}', args).fetchone()['count']
        rows = connection.execute(f'''select ordinal,split,filename,width,height,class_names,has_objects
            from dataset_index_images where {where} order by {order} limit %s offset %s''', [*args, limit, offset]).fetchall()
    return {'images': [{'ordinal': item['ordinal'], 'split': item['split'], 'filename': item['filename'],
                        'width': item['width'], 'height': item['height'], 'classes': item['class_names'],
                        'hasObjects': item['has_objects']} for item in rows], 'total': total, 'offset': offset, 'limit': limit}


def _image(subject, dataset_id, ordinal):
    dataset = _ready(subject, dataset_id)
    with registry._connect() as connection:
        row = connection.execute('''select * from dataset_index_images where dataset_version_id=%s and generation=%s and ordinal=%s''',
                                 (dataset_id, dataset['active_generation'], ordinal)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Image not found.', status=404)
    return dataset, row


@router.get('/{dataset_id}/images/{ordinal}')
def image_detail(dataset_id: uuid.UUID, ordinal: int, request: Request):
    _, row = _image(owner(request)['id'], dataset_id, ordinal)
    return {'ordinal': row['ordinal'], 'filename': row['filename'], 'split': row['split'],
            'width': row['width'], 'height': row['height'], 'annotations': row['annotations']}


@router.get('/{dataset_id}/images/{ordinal}/{variant}')
def image_file(dataset_id: uuid.UUID, ordinal: int, variant: Literal['source', 'thumbnail', 'mask'], request: Request):
    dataset, row = _image(owner(request)['id'], dataset_id, ordinal)
    field = {'source': 'source_path', 'thumbnail': 'thumbnail_path', 'mask': 'mask_path'}[variant]
    if not row[field]:
        raise PublicError('NOT_FOUND', 'Image file not found.', status=404)
    root = registered_storage_path(DATASET_DIR, dataset['storage_path'])
    path = contained_path(root, row[field])
    if not path.is_file():
        raise PublicError('RESULT_UNAVAILABLE', 'This image file is unavailable.', status=410)
    if variant == 'source':
        return viewer_image(path)
    return FileResponse(path, headers={'Cache-Control': 'private, no-store'})


@router.get('/{dataset_id}/duplicates')
def duplicates(dataset_id: uuid.UUID, request: Request, offset: int = Query(default=0, ge=0),
               limit: int = Query(default=20, ge=1, le=48)):
    dataset = _ready(owner(request)['id'], dataset_id)
    with registry._connect() as connection:
        total = connection.execute('''select count(*) as count from (select pixel_sha256 from dataset_index_images
            where dataset_version_id=%s and generation=%s group by pixel_sha256 having count(distinct split)>1) groups''',
            (dataset_id, dataset['active_generation'])).fetchone()['count']
        groups = connection.execute('''select pixel_sha256,count(*) as count from dataset_index_images
            where dataset_version_id=%s and generation=%s group by pixel_sha256 having count(distinct split)>1
            order by count(*) desc,pixel_sha256 limit %s offset %s''',
            (dataset_id, dataset['active_generation'], limit, offset)).fetchall()
        result = []
        for group in groups:
            items = connection.execute('''select ordinal,filename,split from dataset_index_images where dataset_version_id=%s
                and generation=%s and pixel_sha256=%s order by split,filename limit 48''',
                (dataset_id, dataset['active_generation'], group['pixel_sha256'])).fetchall()
            result.append({'hash': group['pixel_sha256'], 'count': group['count'], 'images': [dict(item) for item in items]})
    return {'groups': result, 'total': total, 'offset': offset, 'limit': limit}


@router.get('/{dataset_id}/duplicates/{pixel_hash}')
def duplicate_images(dataset_id: uuid.UUID, pixel_hash: str, request: Request, offset: int = Query(default=0, ge=0),
                     limit: int = Query(default=48, ge=1, le=48)):
    if not re.fullmatch(r'[0-9a-f]{64}', pixel_hash):
        raise PublicError('NOT_FOUND', 'Duplicate group not found.', status=404)
    dataset = _ready(owner(request)['id'], dataset_id)
    with registry._connect() as connection:
        group = connection.execute('''select count(*) as count,count(distinct split) as split_count from dataset_index_images
            where dataset_version_id=%s and generation=%s and pixel_sha256=%s''',
            (dataset_id, dataset['active_generation'], pixel_hash)).fetchone()
        if group['split_count'] < 2:
            raise PublicError('NOT_FOUND', 'Duplicate group not found.', status=404)
        rows = connection.execute('''select ordinal,filename,split from dataset_index_images where dataset_version_id=%s
            and generation=%s and pixel_sha256=%s order by split,filename,ordinal limit %s offset %s''',
            (dataset_id, dataset['active_generation'], pixel_hash, limit, offset)).fetchall()
    return {'images': [dict(row) for row in rows], 'total': group['count'], 'offset': offset, 'limit': limit}
