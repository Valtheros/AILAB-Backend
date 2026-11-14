"""Content manifests for immutable dataset versions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MANIFEST_NAME = '.ailab_source_manifest.json'
GENERATED_DIRECTORIES = {'.ailab_exports', '.ailab_index', '.locks'}


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = []
    for directory, subdirs, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        subdirs[:] = sorted(name for name in subdirs if base != root or name not in GENERATED_DIRECTORIES)
        if any((base / name).is_symlink() for name in subdirs):
            raise ValueError('Dataset contains a symbolic link.')
        for name in sorted(filenames):
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Dataset contains an unsupported file.')
            if base == root and name in {MANIFEST_NAME, '.ailab_dataset.json'}:
                continue
            before = path.stat()
            with path.open('rb') as stream:
                digest = sha256_stream(stream)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Dataset changed while its content was being recorded. Please retry.')
            files.append({'path': path.relative_to(root).as_posix(), 'size': before.st_size, 'sha256': digest})
    files.sort(key=lambda item: item['path'])
    return {
        'format': 1,
        'files': files,
        'classes': metadata.get('classes', []),
        'formats': metadata.get('formats', []),
        'tasks': metadata.get('tasks', []),
        'label_schema': metadata.get('label_schema'),
    }


def encode_manifest(manifest: dict[str, Any]) -> tuple[bytes, str]:
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return encoded, hashlib.sha256(encoded).hexdigest()


def write_manifest(root: Path, manifest: dict[str, Any]) -> str:
    encoded, digest = encode_manifest(manifest)
    (root / MANIFEST_NAME).write_bytes(encoded)
    return digest


def verify_source(root: Path, expected: str | None) -> bool:
    if not expected:
        return False
    path = root / MANIFEST_NAME
    if path.is_symlink():
        raise ValueError('Dataset manifest is not a regular file.')
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError('Dataset manifest no longer matches its recorded version.')
    manifest = json.loads(encoded)
    if encode_manifest(source_manifest(root, manifest))[1] != expected:
        raise ValueError('Dataset changed after this version was created.')
    return True
