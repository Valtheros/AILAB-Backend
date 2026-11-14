"""Read-only, recursive owner inventory. Ambiguous media is retained and reported."""
from __future__ import annotations

from collections import Counter
import hashlib
import os
import re
from pathlib import Path, PurePosixPath
import stat


def owner_hash(owner):
    return hashlib.sha256(owner.encode('utf-8')).hexdigest()[:32]


def ownership_roots(owners, datasets, runs, jobs, operations, label_projects):
    owners = set(owners)
    roots = {}

    def add(path, owner):
        parts = PurePosixPath(path).parts
        if owner not in owners or not parts or parts[0] not in {'dataset', 'runs', 'storage', 'label-studio'} or '..' in parts:
            raise ValueError('Unverified storage ownership: ' + path)
        if path in roots and roots[path] != owner:
            raise ValueError('Conflicting storage ownership: ' + path)
        for existing, assigned in roots.items():
            if assigned != owner and (path.startswith(existing + '/') or existing.startswith(path + '/')):
                raise ValueError('Overlapping storage ownership: ' + path)
        roots[path] = owner

    for owner in owners:
        key = owner_hash(owner)
        for namespace in ('dataset/.owners', 'dataset/.annotations', 'dataset/.annotation-uploads',
                          'dataset/.label-studio-uploads', 'storage/.uploads', 'label-studio/.uploads'):
            add(namespace + '/' + key, owner)
    for namespace, records in (('dataset', datasets), ('runs', runs)):
        for row in records:
            if not row.get('storage_path'):
                continue
            path = PurePosixPath(row['storage_path']).relative_to('/app/' + namespace)
            if not path.parts or path.parts[0] in {'.compute', '.locks'}:
                raise ValueError('Invalid registered resource path')
            add(namespace + '/' + path.as_posix(), row['owner_user_id'])
    for row in jobs:
        add('runs/.compute/' + str(row['id']), row['owner_user_id'])
    for row in operations:
        for kind in ('work', 'previews'):
            add('dataset/.label-studio-' + kind + '/' + str(row['id']), row['owner_user_id'])
        if row.get('upload_path'):
            path = PurePosixPath(row['upload_path']).relative_to('/app/dataset/.label-studio-uploads')
            add('dataset/.label-studio-uploads/' + path.as_posix(), row['owner_user_id'])
    for row in label_projects:
        add('label-studio/media/upload/' + str(row['id']), row['owner'])
    return roots


def category(path):
    parts = PurePosixPath(path).parts
    if parts[0] == 'label-studio':
        return 'temporary' if '.uploads' in parts else 'annotation'
    if parts[0] == 'runs':
        if '.compute' in parts:
            return 'temporary'
        if 'evaluations' in parts or '.evaluations' in parts:
            return 'evaluations'
        return 'runs'
    if any(name in parts for name in ('.ailab_index', '.ailab_cache', '.ailab_exports')):
        return 'cache'
    if parts[0] == 'storage' or any(p.startswith(('.staged', '.upload', '.label-studio-', '.annotation-work')) for p in parts):
        return 'temporary'
    if '.annotations' in parts or '.annotation-uploads' in parts:
        return 'annotation'
    return 'datasets'


def scan(root, roots, *, hashes=True, attributes=None):
    root = Path(root)
    device = root.stat().st_dev
    records, issues, inodes, links = [], [], {}, Counter()
    ordered = sorted(roots, key=len, reverse=True)
    totals = {}
    for namespace in ('dataset', 'runs', 'storage', 'label-studio'):
        base = root / namespace
        if base.is_symlink() or not base.is_dir():
            raise ValueError('Invalid storage namespace: ' + namespace)
        for current, dirs, files in os.walk(base, followlinks=False):
            if Path(current).stat().st_dev != device:
                issues.append({'path': Path(current).relative_to(root).as_posix(), 'reason': 'Filesystem boundary'})
                dirs[:] = []
                continue
            for name in ['.', *files, *[name for name in dirs if (Path(current) / name).is_symlink()]]:
                path = Path(current) if name == '.' else Path(current) / name
                relative = path.relative_to(root).as_posix()
                info = path.lstat()
                if info.st_dev != device or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    issues.append({'path': relative, 'reason': 'Link, special file or filesystem boundary'})
                    continue
                owner = next((roots[p] for p in ordered if relative == p or relative.startswith(p + '/')), None)
                # Only known service metadata and empty unassigned directories are exempt.
                system_lock = (re.fullmatch(r'dataset/\.locks/dataset-[0-9a-f]{64}\.lock', relative)
                               and info.st_size <= 20 and path.read_bytes().isdigit())
                system_file = relative == 'label-studio/.env' or system_lock
                if owner is None and stat.S_ISREG(info.st_mode) and not system_file:
                    issues.append({'path': relative, 'reason': 'File has no verified owner'})
                identity = (info.st_dev, info.st_ino)
                if stat.S_ISREG(info.st_mode):
                    links[identity] += 1
                    if identity in inodes and inodes[identity][0] != owner:
                        issues.append({'path': relative, 'reason': 'Hard link crosses owners'})
                fresh = identity not in inodes
                inodes[identity] = (owner, info.st_nlink, relative, stat.S_ISREG(info.st_mode))
                record = {'path': relative, 'owner': owner, 'inode': info.st_ino, 'size': info.st_size,
                          'mtimeNs': info.st_mtime_ns, 'mode': info.st_mode, 'blocks': info.st_blocks,
                          'links': info.st_nlink}
                if owner and fresh:
                    group = totals.setdefault(owner, {})
                    bucket = category(relative)
                    group[bucket] = group.get(bucket, 0) + info.st_blocks * 512
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != identity:
                        raise ValueError('Storage changed during inventory: ' + relative)
                    if attributes:
                        attr = attributes(fd)
                        record['project'], record['inherit'] = attr[3], bool(attr[0] & 0x200)
                    if hashes and stat.S_ISREG(info.st_mode):
                        digest = hashlib.sha256()
                        while chunk := os.read(fd, 1024**2):
                            digest.update(chunk)
                        record['sha256'] = digest.hexdigest()
                    after = os.fstat(fd)
                    if (after.st_size, after.st_mtime_ns, after.st_nlink) != (info.st_size, info.st_mtime_ns, info.st_nlink):
                        raise ValueError('Storage changed during inventory: ' + relative)
                finally:
                    os.close(fd)
                records.append(record)
    for identity, (_owner, expected, path, regular) in inodes.items():
        if regular and links[identity] != expected:
            issues.append({'path': path, 'reason': 'Hard link escapes inventoried storage'})
    return {'files': records, 'issues': issues, 'breakdown': totals}
