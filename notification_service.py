"""Persistent, idempotent user-visible operation events."""
from __future__ import annotations


def emit(connection, owner_id, event_key, kind, status, target_path):
    if not target_path.startswith('/') or target_path.startswith('//'):
        raise ValueError('Notification target must be an application path.')
    connection.execute('''insert into notifications(owner_user_id,event_key,kind,status,target_path)
        values(%s,%s,%s,%s,%s) on conflict(event_key) do nothing''',
        (owner_id, event_key, kind, status, target_path))
