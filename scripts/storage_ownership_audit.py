"""Report ambiguous top-level storage ownership without modifying data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dataset_storage import owner_storage_key
from resource_repository import resource_repository as registry


def _assign(mapping, name, owner, issues, category):
    if not owner:
        issues.append({'category': category, 'path': name, 'reason': 'missing owner'})
        return
    if name in mapping and mapping[name] != owner:
        mapping[name] = None
        issues.append({'category': category, 'path': name, 'reason': 'conflicting owners'})
    else:
        mapping[name] = owner


def _inspect_children(root, mapping, category, issues, *, optional=False):
    if root.is_symlink():
        issues.append({'category': category, 'path': str(root), 'reason': 'symbolic link needs review'})
        return 0
    if optional and not root.exists():
        return 0
    try:
        with os.scandir(root) as entries:
            items = list(entries)
    except OSError as exc:
        issues.append({'category': category, 'path': str(root), 'reason': f'cannot read directory: {exc.strerror}'})
        return 0
    assigned = 0
    for entry in items:
        path = str(root / entry.name)
        if entry.is_symlink():
            issues.append({'category': category, 'path': path, 'reason': 'symbolic link needs review'})
        elif entry.name not in mapping or not mapping[entry.name]:
            issues.append({'category': category, 'path': path, 'reason': 'no unambiguous owner'})
        else:
            assigned += 1
    return assigned


def audit(dataset_root, runs_root, label_upload_root, *, owners, datasets, runs, projects, jobs):
    """Inspect root associations only; nested links and concurrent writes need a separate audit."""
    dataset_root, runs_root, label_upload_root = (Path(path).resolve(strict=True)
                                                   for path in (dataset_root, runs_root, label_upload_root))
    issues = []
    owners = set(owners)
    owner_map, dataset_top, run_top, project_map, job_map = {}, {}, {}, {}, {}
    for owner in owners:
        if not owner:
            issues.append({'category': 'owner-key', 'path': '', 'reason': 'missing owner'})
            continue
        _assign(owner_map, owner_storage_key(owner), owner, issues, 'owner-key')
    for item in datasets:
        try:
            if item['owner_user_id'] not in owners:
                raise ValueError('dataset owner is not in the user table')
            path = Path(os.path.abspath(item['storage_path']))
            parts = path.relative_to(dataset_root).parts
            if not parts:
                raise ValueError('storage path is the dataset root')
            if parts[0] == '.owners':
                if len(parts) < 2 or owner_map.get(parts[1]) != item['owner_user_id']:
                    raise ValueError('dataset path and owner do not match')
            else:
                _assign(dataset_top, parts[0], item['owner_user_id'], issues, 'dataset')
        except (KeyError, TypeError, ValueError) as exc:
            issues.append({'category': 'dataset', 'path': str(item.get('storage_path')), 'reason': str(exc)})
    for item in runs:
        if not item.get('storage_path'):
            continue
        try:
            if item['owner_user_id'] not in owners:
                raise ValueError('run owner is not in the user table')
            path = Path(os.path.abspath(item['storage_path']))
            parts = path.relative_to(runs_root).parts
            if not parts or parts[0] == '.compute':
                raise ValueError('invalid run storage path')
            _assign(run_top, parts[0], item['owner_user_id'], issues, 'run')
        except (KeyError, TypeError, ValueError) as exc:
            issues.append({'category': 'run', 'path': str(item.get('storage_path')), 'reason': str(exc)})
    for item in projects:
        owner = item['owner_user_id'] if item['owner_user_id'] in owners else None
        _assign(project_map, str(item['ls_project_id']), owner, issues, 'label-studio')
    for item in jobs:
        owner = item['owner_user_id'] if item['owner_user_id'] in owners else None
        _assign(job_map, str(item['id']), owner, issues, 'compute')

    counts = {}
    for name in ('.owners', '.annotations', '.annotation-uploads'):
        counts[name] = _inspect_children(dataset_root / name, owner_map, name, issues, optional=True)
    counts['label-studio'] = _inspect_children(label_upload_root, project_map, 'label-studio', issues)
    counts['compute'] = _inspect_children(runs_root / '.compute', job_map, 'compute', issues, optional=True)
    if '.compute' in run_top:
        issues.append({'category': 'runs', 'path': str(runs_root / '.compute'), 'reason': 'run path overlaps system storage'})
    run_top['.compute'] = 'system'
    counts['runs'] = _inspect_children(runs_root, run_top, 'runs', issues)
    for name in ('.owners', '.annotations', '.annotation-uploads', '.locks'):
        if name in dataset_top:
            issues.append({'category': 'datasets', 'path': str(dataset_root / name), 'reason': 'dataset path overlaps system storage'})
        dataset_top[name] = 'system'
    for name in ('.staged', '.annotation-work', '.label-studio-uploads', '.label-studio-work', '.label-studio-previews'):
        path = dataset_root / name
        if path.is_dir() and not path.is_symlink():
            try:
                if not any(path.iterdir()):
                    dataset_top[name] = 'empty'
            except OSError:
                pass
    counts['datasets'] = _inspect_children(dataset_root, dataset_top, 'datasets', issues)
    return {'counts': counts, 'issues': issues, 'ownershipVerified': False,
            'note': 'Top-level associations only. Nested files, hard links, temporary writers and concurrent changes require separate verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--runs-root', required=True)
    parser.add_argument('--label-upload-root', required=True)
    args = parser.parse_args()
    with registry._connect() as connection:
        connection.execute('set transaction isolation level repeatable read read only')
        owners = [row['id'] for row in connection.execute('select id from "user"').fetchall()]
        datasets = connection.execute('select owner_user_id,storage_path from datasets').fetchall()
        runs = connection.execute('select owner_user_id,storage_path from training_tasks').fetchall()
        projects = connection.execute('select owner_user_id,ls_project_id from label_studio_projects').fetchall()
        jobs = connection.execute('select owner_user_id,id from compute_jobs').fetchall()
    result = audit(args.dataset_root, args.runs_root, args.label_upload_root, owners=owners, datasets=datasets,
                   runs=runs, projects=projects, jobs=jobs)
    print(json.dumps(result, indent=2))
    return 1 if result['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
