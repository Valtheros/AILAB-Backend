from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field, StrictInt

from api_errors import PublicError
from label_studio_api import owner
from resource_repository import resource_repository as registry
from storage_service import enabled, ledger


router = APIRouter(prefix='/api/storage')


def require_enabled():
    if not enabled():
        raise PublicError('STORAGE_UNAVAILABLE', 'Storage accounting is not available yet.', status=503)


def admin(request):
    subject = owner(request)['id']
    try:
        registry.assert_admin(subject)
    except PermissionError as exc:
        raise PublicError('ADMIN_REQUIRED', 'Administrator access is required.', status=403) from exc
    require_enabled()


def usage(subject):
    result = ledger.usage(subject)
    with registry._connect() as c:
        row = c.execute('select breakdown,reconciled_at from storage_accounts where owner_user_id=%s', (subject,)).fetchone()
    ratio = result['usedBytes'] / result['limitBytes']
    return {**result, 'warning': 'full' if ratio >= 1 else 'critical' if ratio >= .95 else 'warning' if ratio >= .8 else None,
            'breakdown': row['breakdown'], 'reconciledAt': row['reconciled_at']}


@router.get('')
def storage(request: Request):
    subject = owner(request)['id']
    require_enabled()
    ledger.prepare_account(subject)
    return usage(subject)


@router.get('/admin')
def users(request: Request, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    admin(request)
    with registry._connect() as c:
        total = c.execute('select count(*) as count from "user"').fetchone()['count']
        rows = c.execute('''select u.id,u.email,a.state,a.limit_bytes from "user" u
            left join storage_accounts a on a.owner_user_id=u.id order by u.id limit %s offset %s''', (limit, offset)).fetchall()
    return {'total': total, 'users': [{'id': row['id'], 'email': row['email'], 'state': row['state'] or 'pending',
        **(usage(row['id']) if row['state'] == 'ready' else {'limitBytes': row['limit_bytes'] or 20 * 1024**3})} for row in rows]}


class LimitUpdate(BaseModel):
    limitGiB: StrictInt = Field(ge=1, le=1048576)


@router.put('/admin/{subject}')
def update_limit(subject: str, payload: LimitUpdate, request: Request):
    admin(request)
    ledger.set_limit(subject, payload.limitGiB * 1024**3)
    return usage(subject)
