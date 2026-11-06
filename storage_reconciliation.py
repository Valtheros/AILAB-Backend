"""Periodic category snapshots; the kernel quota remains the live usage authority."""
import json
import logging
import os
from pathlib import Path
import stat

from storage_inventory import category
from storage_project_quota import _attributes


def breakdown(root, projects, attributes=_attributes):
    root = Path(root)
    device = root.stat().st_dev
    totals = {owner: {} for owner in projects.values()}
    seen = set()
    for namespace in ('dataset', 'runs', 'storage', 'label-studio'):
        base = root / namespace
        if base.is_symlink() or not base.is_dir():
            raise ValueError('Missing or unsafe storage namespace')
        for current, dirs, files in os.walk(base, followlinks=False):
            for name in ['.', *files]:
                path = Path(current) if name == '.' else Path(current) / name
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
                except FileNotFoundError:
                    continue
                try:
                    info = os.fstat(fd)
                    if info.st_dev != device or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                        raise ValueError('Unexpected filesystem boundary or file type')
                    identity = (info.st_dev, info.st_ino)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    owner = projects.get(attributes(fd)[3])
                    if owner:
                        bucket = category(path.relative_to(root).as_posix())
                        totals[owner][bucket] = totals[owner].get(bucket, 0) + info.st_blocks * 512
                finally:
                    os.close(fd)
            if any((Path(current) / name).is_symlink() for name in dirs):
                raise ValueError('Unexpected symbolic link in user storage')
    return totals


def reconcile():
    from resource_repository import resource_repository as registry
    from storage_service import enabled, ledger
    if not enabled():
        return
    with registry._connect() as connection:
        rows = connection.execute("select owner_user_id,project_id from storage_accounts where state='ready'").fetchall()
    totals = breakdown(Path(os.environ.get('STORAGE_INVENTORY_ROOT', '/userdata-readonly')),
                       {row['project_id']: row['owner_user_id'] for row in rows})
    with registry._connect() as connection:
        for owner, values in totals.items():
            connection.execute("update storage_accounts set breakdown=%s::jsonb,reconciled_at=now() where owner_user_id=%s and state='ready'",
                               (json.dumps(values), owner))
    for owner in totals:
        ledger.trim_idle(owner)
    logging.getLogger(__name__).info('Storage category snapshots updated for %s accounts', len(totals))
