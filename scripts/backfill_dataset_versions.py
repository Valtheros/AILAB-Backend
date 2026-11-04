"""Register existing datasets as versions without changing their paths or run history."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from dataset_storage import registered_storage_path
from dataset_versioning import MANIFEST_NAME, encode_manifest, source_manifest, write_manifest
from resource_repository import resource_repository as registry
from settings import DATASET_DIR


def backfill(apply: bool = False) -> dict[str, int]:
    if not registry.enabled:
        raise RuntimeError('DATABASE_URL is required')
    with registry._connect() as connection:
        rows = connection.execute(
            '''select id,owner_user_id,slug,storage_path,status from datasets
               where owner_user_id is not null order by created_at,id'''
        ).fetchall()
    totals = {'scanned': 0, 'updated': 0, 'missing': 0}
    for original in rows:
        totals['scanned'] += 1
        owner, slug = original['owner_user_id'], original['slug']
        path = registered_storage_path(DATASET_DIR, original['storage_path'])
        with registry.dataset_guard(owner, slug) as connection:
            row = connection.execute('select * from datasets where id=%s and owner_user_id=%s for update',
                                     (original['id'], owner)).fetchone()
            if not row:
                continue
            if row.get('collection_id') and (row.get('content_sha256') or not path.is_dir()):
                continue
            metadata = row.get('metadata') or {}
            digest = None
            document = None
            if path.is_dir():
                document = source_manifest(path, metadata)
                _, digest = encode_manifest(document)
                existing = path / MANIFEST_NAME
                if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Existing dataset manifest differs for {row["id"]}; no files were changed.')
            else:
                totals['missing'] += 1
            totals['updated'] += 1
            if not apply:
                continue
            if document is not None and not (path / MANIFEST_NAME).exists():
                write_manifest(path, document)
            operation = metadata.get('label_studio_operation')
            linked = connection.execute(
                '''select o.project_id,o.version,p.name from label_studio_operations o
                   join label_studio_projects p on p.id=o.project_id
                   where o.id::text=%s and o.owner_user_id=%s and o.version is not null''',
                (str(operation or ''), owner),
            ).fetchone()
            if linked and not row.get('collection_id'):
                registry.attach_label_studio_version(owner, linked['project_id'], linked['name'], linked['version'],
                                                    row['id'], path, digest, connection)
            elif not row.get('collection_id'):
                collection = registry.reserve_dataset_version(owner, slug, None, connection)
                connection.execute('''update datasets set collection_id=%s,version=%s,content_sha256=%s,
                    source_manifest_path=%s where id=%s''',
                    (collection['id'], collection['reserved_version'], digest,
                     str(path / MANIFEST_NAME) if digest else None, row['id']))
            elif digest:
                connection.execute('update datasets set content_sha256=%s,source_manifest_path=%s where id=%s',
                                   (digest, str(path / MANIFEST_NAME), row['id']))
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='write version metadata and content manifests')
    args = parser.parse_args()
    print(('APPLY' if args.apply else 'DRY RUN') + ': ' + str(backfill(args.apply)))
