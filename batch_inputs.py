"""Validation helpers for bounded batch image uploads."""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path, PurePosixPath

from dataset_utils import IMAGE_EXTENSIONS
from security_utils import _validate_archive_member

MAX_BATCH_IMAGES = 100
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH_BYTES = 1024 * 1024 * 1024


def extract_batch_zip(archive_path: Path, directory: Path) -> list[dict]:
    entries = []
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 1000:
            raise ValueError('ZIP contains too many entries.')
        images = []
        total = 0
        for member in members:
            name = member.filename.replace('\\', '/')
            if '..' in PurePosixPath(name).parts or '\x00' in name:
                raise ValueError('ZIP contains an unsafe path.')
            _validate_archive_member(member, directory)
            if member.is_dir() or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            images.append(member)
            total += member.file_size
            if len(images) > MAX_BATCH_IMAGES or total > MAX_BATCH_BYTES:
                raise ValueError('Batch exceeds the image count or expanded size limit.')
        if not images:
            raise ValueError('ZIP contains no supported images.')
        inputs = directory / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(images):
            entry = {'name': member.filename.replace('\\', '/')[:512]}
            destination = inputs / str(index)
            if member.file_size > MAX_IMAGE_BYTES:
                entry['error'] = 'Image exceeds the 10 MiB limit.'
            else:
                try:
                    size = 0
                    digest = hashlib.sha256()
                    with archive.open(member) as source, destination.open('wb') as output:
                        while chunk := source.read(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_IMAGE_BYTES:
                                raise ValueError('Image exceeds the 10 MiB limit.')
                            output.write(chunk)
                            digest.update(chunk)
                    entry['bytes'] = size
                    entry['sha256'] = digest.hexdigest()
                except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
                    destination.unlink(missing_ok=True)
                    entry['error'] = 'Image could not be read from the ZIP.'
            entries.append(entry)
    return entries


def safe_image_name(filename: str) -> str:
    name = os.path.basename(filename.replace('\\', '/'))[:255]
    if not name or '\x00' in name:
        return 'image'
    return name
