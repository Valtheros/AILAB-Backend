"""Offline, idempotent ownership backfill. Refuses unknown files and changed inventories."""
from __future__ import annotations

import argparse
import fcntl
import json
import hashlib
import os
from pathlib import Path
import stat

from storage_inventory import ownership_roots, scan
from storage_operator import QuotaOperator, open_relative
from storage_project_quota import _attributes, _FSXATTR, _FS_IOC_FSSETXATTR, _FS_XFLAG_PROJINHERIT


def metadata(connection, label_inventory):
    if label_inventory.get('issues'):
        raise ValueError('Label Studio has unverified upload ownership')
    owners = [r['id'] for r in connection.execute('select id from "user" order by id').fetchall()]
    def rows(sql):
        return [dict(row) for row in connection.execute(sql).fetchall()]
    roots = ownership_roots(owners, rows('select owner_user_id,storage_path from datasets'),
        rows('select owner_user_id,storage_path from training_tasks'),
        rows('select id,owner_user_id from compute_jobs'),
        rows('select id,owner_user_id,upload_path from label_studio_operations'), label_inventory['projects'])
    return owners, roots


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    with temporary.open('w', encoding='utf-8') as file:
        os.chmod(temporary, 0o600)
        json.dump(value, file, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def verify_inventory(root, expected, roots):
    current = scan(root, roots, attributes=_attributes)
    if current['issues']:
        raise ValueError('Storage ownership has unresolved issues: ' + json.dumps(current['issues'][:10]))
    def identity(rows):
        return {r['path']: {k: v for k, v in r.items() if k not in {'project', 'inherit'}} for r in rows}
    if identity(current['files']) != identity(expected['files']):
        raise ValueError('Files changed after the inventory. Generate a new plan while writers are stopped.')
    return current


def tag_records(root, records, projects, *, restore=False):
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for row in records:
            if not row['owner']:
                continue
            target = projects[row['owner']]
            path = Path(row['path'])
            parent = open_relative(root_fd, path.parent.as_posix(), project_id=target)
            try:
                fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            finally:
                os.close(parent)
            try:
                info = os.fstat(fd)
                if (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns) != (
                        os.fstat(root_fd).st_dev, row['inode'], row['mode'], row['size'], row['mtimeNs']):
                    raise ValueError('File changed before tagging: ' + row['path'])
                attrs = _attributes(fd)
                if attrs[3] not in {row['project'], target} or row['project'] not in {0, target}:
                    raise ValueError('File is assigned to a different quota owner: ' + row['path'])
                project = row['project'] if restore else target
                inherit = row['inherit'] if restore else stat.S_ISDIR(info.st_mode)
                attrs[3] = project
                attrs[0] = (attrs[0] | _FS_XFLAG_PROJINHERIT) if inherit else (attrs[0] & ~_FS_XFLAG_PROJINHERIT)
                fcntl.ioctl(fd, _FS_IOC_FSSETXATTR, bytearray(_FSXATTR.pack(*attrs)), True)
                actual = _attributes(fd)
                if actual[3] != project or bool(actual[0] & _FS_XFLAG_PROJINHERIT) != inherit:
                    raise RuntimeError('Filesystem did not retain quota ownership')
            finally:
                os.close(fd)
    finally:
        os.close(root_fd)


def apply_plan(connection, root, device, plan, journal_path):
    if connection.execute("select 1 from storage_transfers where state in ('copying','committed') limit 1").fetchone():
        raise ValueError('Recover account transfers before ownership backfill')
    if connection.execute("select 1 from compute_jobs where status in ('dispatching','running','stopping','recovery_pending') limit 1").fetchone():
        raise ValueError('Stop active compute jobs before ownership backfill')
    if connection.execute("select 1 from dataset_indexes where status='running' limit 1").fetchone():
        raise ValueError('Stop dataset indexing before ownership backfill')
    if connection.execute("select 1 from label_studio_operations where status='running' limit 1").fetchone():
        raise ValueError('Stop active Label Studio operations before ownership backfill')
    verify_inventory(root, plan, plan['roots'])
    operator = QuotaOperator(root, device)
    projects = {}
    for owner in plan['owners']:
        row = connection.execute('''insert into storage_accounts(owner_user_id) values(%s)
            on conflict(owner_user_id) do update set state='pending' returning *''', (owner,)).fetchone()
        projects[owner] = row['project_id']
        usage = sum(plan['breakdown'].get(owner, {}).values())
        ceiling = max(row['limit_bytes'], usage + 1024**2)
        connection.execute('update storage_accounts set enforced_limit_bytes=%s where owner_user_id=%s', (ceiling, owner))
        operator.dispatch({'operation': 'provision', 'owner': owner, 'projectId': row['project_id']})
        operator.dispatch({'operation': 'limit', 'owner': owner, 'projectId': row['project_id'], 'limitBytes': ceiling})
    journal = {'state': 'applying', 'projects': projects,
               'planSha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()}
    save(journal_path, journal)
    tag_records(root, plan['files'], projects)
    current = verify_inventory(root, plan, plan['roots'])
    for row in current['files']:
        if row['owner'] and (row['project'] != projects[row['owner']] or row['inherit'] != stat.S_ISDIR(row['mode'])):
            raise RuntimeError('Post-backfill ownership verification failed')
    for owner, project in projects.items():
        usage = operator.dispatch({'operation': 'measure', 'owner': owner, 'projectId': project})
        ceiling = max(1024, ((usage['usedBytes'] + 1023) // 1024) * 1024)
        operator.dispatch({'operation': 'limit', 'owner': owner, 'projectId': project, 'limitBytes': ceiling})
        connection.execute('''update storage_accounts set enforced_limit_bytes=%s,state='ready',breakdown=%s::jsonb,
            reconciled_at=now(),updated_at=now() where owner_user_id=%s''',
            (ceiling, json.dumps(current['breakdown'].get(owner, {})), owner))
    save(journal_path, {**journal, 'state': 'verified'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'apply', 'verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', type=Path)
    parser.add_argument('--label-inventory', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--maintenance-confirmed', action='store_true')
    args = parser.parse_args()
    from resource_repository import resource_repository as registry
    from storage_reservations import _GLOBAL_LOCK
    with registry._connect() as connection:
        connection.autocommit = True
        connection.execute('select pg_advisory_lock(%s)', (_GLOBAL_LOCK,))
        owners, roots = metadata(connection, json.loads(args.label_inventory.read_text()))
        if args.action == 'plan':
            result = {**scan(args.root, roots, attributes=_attributes), 'owners': owners, 'roots': roots}
            save(args.plan, result)
            print(json.dumps({'files': len(result['files']), 'owners': len(owners), 'issues': result['issues'],
                              'breakdown': result['breakdown']}))
            return 1 if result['issues'] else 0
        plan = json.loads(args.plan.read_text())
        if plan['owners'] != owners or plan['roots'] != roots or plan['issues']:
            raise ValueError('Ownership records changed or the plan contains unresolved issues')
        if args.action == 'verify':
            verify_inventory(args.root, plan, roots)
            print('File contents and registered ownership are unchanged; use apply for quota-tag verification.')
        else:
            if not args.maintenance_confirmed or args.device is None:
                raise ValueError('Apply requires a stopped-writer maintenance window and explicit quota device')
            apply_plan(connection, args.root, args.device, plan, args.plan.with_suffix('.journal.json'))
            print('Owner backfill, content hashes and filesystem quotas verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
