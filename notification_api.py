from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from api_errors import PublicError
from label_studio_api import owner
from resource_repository import resource_repository as registry

router = APIRouter(prefix='/api/notifications')


def target_path(row):
    target = row['target_path']
    if row['kind'] not in {'evaluate', 'predict', 'batch_predict'} or not target.startswith('/tasks/'):
        return target
    parts = row['event_key'].split(':')
    try:
        job_id = str(uuid.UUID(parts[1])) if len(parts) == 3 and parts[0] == 'compute' else None
    except ValueError:
        job_id = None
    if not job_id:
        return target
    separator = '&' if '?' in target else '?'
    if row['kind'] == 'evaluate':
        return f'{target}{separator}evaluationJob={job_id}'
    return f"{target}{separator}predictionJob={job_id}&predictionKind={row['kind']}"


@router.get('')
def list_notifications(request: Request):
    subject = owner(request)['id']
    with registry._connect() as connection:
        rows = connection.execute('''select id,event_key,kind,status,target_path,created_at,read_at from notifications
            where owner_user_id=%s and created_at>=now()-interval '90 days' order by created_at desc limit 50''', (subject,)).fetchall()
        unread = connection.execute('''select count(*) as count from notifications where owner_user_id=%s
            and read_at is null and created_at>=now()-interval '90 days' ''', (subject,)).fetchone()['count']
    return {'notifications': [{'id': str(row['id']), 'kind': row['kind'], 'status': row['status'],
                               'targetPath': target_path(row), 'createdAt': row['created_at'],
                               'readAt': row['read_at']} for row in rows], 'unreadCount': unread}


@router.post('/read-all')
def mark_all_read(request: Request):
    with registry._connect() as connection:
        connection.execute('update notifications set read_at=now() where owner_user_id=%s and read_at is null',
                           (owner(request)['id'],))
    return {'ok': True}


@router.post('/{notification_id}/read')
def mark_read(notification_id: uuid.UUID, request: Request):
    subject = owner(request)['id']
    with registry._connect() as connection:
        row = connection.execute('''update notifications set read_at=coalesce(read_at,now())
            where id=%s and owner_user_id=%s returning id''', (notification_id, subject)).fetchone()
    if not row:
        raise PublicError('NOT_FOUND', 'Notification not found.', status=404)
    return {'ok': True}
