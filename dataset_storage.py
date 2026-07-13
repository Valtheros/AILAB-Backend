from __future__ import annotations

import hashlib
from pathlib import Path

from security_utils import contained_path, validate_slug


OWNER_DATASETS_DIR = ".owners"


def owner_storage_key(owner_id: str) -> str:
    if not owner_id:
        raise ValueError("Dataset owner is required")
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:32]


def owner_dataset_root(dataset_root: Path, owner_id: str) -> Path:
    return contained_path(dataset_root, OWNER_DATASETS_DIR, owner_storage_key(owner_id))


def owner_dataset_path(dataset_root: Path, owner_id: str, dataset_slug: str) -> Path:
    validate_slug(dataset_slug, "dataset name")
    return contained_path(owner_dataset_root(dataset_root, owner_id), dataset_slug)


def dataset_lock_name(owner_id: str, dataset_slug: str) -> str:
    digest = hashlib.sha256(f"{owner_id}\0{dataset_slug}".encode("utf-8")).hexdigest()
    return f"dataset-{digest}"


def registered_storage_path(dataset_root: Path, raw_path: str | Path) -> Path:
    return contained_path(dataset_root, Path(raw_path))
