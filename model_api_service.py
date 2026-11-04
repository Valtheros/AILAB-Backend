"""Model endpoint identities and fail-closed public API access."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
from pathlib import Path

from redis import Redis

from api_errors import PublicError
from dataset_versioning import sha256_stream
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import REDIS_URL, RUNS_DIR

# Public submissions must remain disabled until all upload and worker paths enforce storage quotas.
STORAGE_ENFORCEMENT_IMPLEMENTED = False


def enabled():
    return STORAGE_ENFORCEMENT_IMPLEMENTED and os.getenv('MODEL_API_ENABLED', 'false').lower() == 'true'


def checkpoint(run_slug: str, model_type: str):
    from detection_inference import _yolo_checkpoint_path
    from inference_service import _checkpoint_path

    root = contained_path(RUNS_DIR, run_slug)
    path = _yolo_checkpoint_path(root) if model_type == 'yolo' else _checkpoint_path(root)
    path = contained_path(root, path)
    before = path.stat()
    with path.open('rb') as stream:
        digest = sha256_stream(stream)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise PublicError('CHECKPOINT_CHANGED', 'The checkpoint changed while being verified. Please retry.', status=409)
    return path.relative_to(root).as_posix(), digest


def authenticate(authorization):
    if not enabled():
        raise PublicError('NOT_FOUND', 'API not found.', status=404)
    scheme, _, token = (authorization or '').partition(' ')
    if scheme.lower() != 'bearer' or not re.fullmatch(r'ailab_[A-Za-z0-9_-]{43}', token):
        raise PublicError('UNAUTHORIZED', 'A valid model API key is required.', status=401)
    digest = hashlib.sha256(token.encode('ascii')).hexdigest()
    with registry._connect() as connection:
        row = connection.execute('''select k.id as key_id,e.id as endpoint_id,e.owner_user_id,e.run_id,e.enabled,
            e.checkpoint_path,e.checkpoint_sha256,t.run_slug,t.model_type,t.task_type,t.status as run_status
            from model_api_keys k join model_endpoints e on e.id=k.endpoint_id
            join training_tasks t on t.id=e.run_id join "user" u on u.id=e.owner_user_id
            where k.key_sha256=%s and k.revoked_at is null and k.expires_at>now()
            and u."emailVerified" and not coalesce(u.banned,false) and t.owner_user_id=e.owner_user_id''', (digest,)).fetchone()
    if not row:
        raise PublicError('UNAUTHORIZED', 'A valid model API key is required.', status=401)
    return dict(row)


def enforce_rate_limit(principal, submitting):
    identity = principal['owner_user_id'] if submitting else str(principal['key_id'])
    digest = hashlib.sha256(identity.encode()).hexdigest()
    key = f'ailab:model-api:{"submit" if submitting else "read"}:{digest}'
    script = """local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],60) end;
        local ttl=redis.call('TTL',KEYS[1]); return {n,ttl}"""
    try:
        client = Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        count, ttl = client.eval(script, 1, key)
    except Exception as exc:
        raise PublicError('SERVICE_UNAVAILABLE', 'API rate limiting is temporarily unavailable.', status=503) from exc
    if int(count) > (20 if submitting else 120):
        raise PublicError('RATE_LIMITED', 'Too many requests. Please retry later.', status=429,
                          details={'retryAfter': max(1, int(ttl))})


def issue_key(connection, endpoint_id, label):
    secret = 'ailab_' + secrets.token_urlsafe(32)
    digest = hashlib.sha256(secret.encode('ascii')).hexdigest()
    row = connection.execute('''insert into model_api_keys(endpoint_id,key_sha256,label) values(%s,%s,%s)
        returning id,label,created_at,expires_at,revoked_at''', (endpoint_id, digest, label)).fetchone()
    return dict(row), secret


def verify_checkpoint(run_slug, model_type, expected_path, expected_hash):
    try:
        path, digest = checkpoint(run_slug, model_type)
    except (OSError, ValueError) as exc:
        raise PublicError('CHECKPOINT_CHANGED', 'The saved model is no longer available.', status=409) from exc
    if path != expected_path or not secrets.compare_digest(digest, expected_hash):
        raise PublicError('CHECKPOINT_CHANGED', 'The saved model changed. Create a new endpoint for this checkpoint.', status=409)
