from __future__ import annotations

import json
import hashlib
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from dataset_storage import owner_dataset_root
from security_utils import contained_path


STAGED_UPLOAD_TTL_SECONDS = 30 * 60
MAX_STAGED_UPLOADS_PER_USER = 3


class StagedUploadStore:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.resolve()
        self.root = contained_path(self.dataset_root, ".staged")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def owner_root(self, owner_id: str) -> Path:
        return contained_path(owner_dataset_root(self.dataset_root, owner_id), ".staged")

    def _roots(self, owner_id: str | None = None) -> list[Path]:
        if owner_id is not None:
            return [self.owner_root(owner_id), self.root]
        roots = [self.root]
        owners = contained_path(self.dataset_root, ".owners")
        if owners.is_dir():
            for directory in owners.iterdir():
                if directory.is_dir() and not directory.is_symlink():
                    roots.append(contained_path(directory, ".staged"))
        return roots

    @staticmethod
    def _validate_token(token: str) -> None:
        if not token or len(token) > 128 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in token):
            raise ValueError("Invalid upload token")

    def cleanup(self, owner_id: str | None = None) -> None:
        with self._lock:
            now = time.time()
            for root in self._roots(owner_id):
                if not root.is_dir():
                    continue
                for item in root.iterdir():
                    if not item.is_dir() or item.is_symlink() or item.name.startswith(".consuming-"):
                        continue
                    manifest = self._read_manifest(item)
                    if not manifest or (owner_id is not None and manifest.get("owner_id") != owner_id):
                        continue
                    try:
                        expired = float(manifest["expires_at"]) <= now
                    except (KeyError, TypeError, ValueError):
                        continue
                    if expired:
                        shutil.rmtree(item, ignore_errors=True)

    def create(self, owner_id: str, dataset_name: str, extracted_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.cleanup(owner_id)
            owned = [
                item for root in self._roots(owner_id) if root.is_dir() for item in root.iterdir()
                if item.is_dir() and not item.is_symlink() and self._read_manifest(item).get("owner_id") == owner_id
            ]
            if len(owned) >= MAX_STAGED_UPLOADS_PER_USER:
                raise ValueError(f"At most {MAX_STAGED_UPLOADS_PER_USER} pending dataset uploads are allowed per user")
            token = secrets.token_urlsafe(32)
            root = self.owner_root(owner_id)
            root.mkdir(parents=True, exist_ok=True)
            stage_dir = contained_path(root, token)
            stage_dir.mkdir(mode=0o700)
            target = contained_path(stage_dir, "dataset")
            extracted_dir.replace(target)
            expires_at = int(time.time()) + STAGED_UPLOAD_TTL_SECONDS
            manifest = {
                "token": token,
                "owner_id": owner_id,
                "dataset_name": dataset_name,
                "expires_at": expires_at,
                "profile": profile,
                "fingerprint": self._fingerprint(target),
            }
            (stage_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            return manifest

    def peek(self, token: str, owner_id: str) -> tuple[dict[str, Any], Path, Path]:
        self._validate_token(token)
        stage_dir = None
        manifest = {}
        for root in self._roots(owner_id):
            candidate = contained_path(root, token)
            candidate_manifest = self._read_manifest(candidate)
            if candidate_manifest.get("token") == token and candidate_manifest.get("owner_id") == owner_id:
                stage_dir, manifest = candidate, candidate_manifest
                break
        if stage_dir is None:
            raise FileNotFoundError("Pending upload was not found")
        if float(manifest.get("expires_at", 0)) <= time.time():
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError("Pending upload has expired. Upload the ZIP again.")
        dataset_dir = contained_path(stage_dir, "dataset")
        if not dataset_dir.is_dir():
            raise FileNotFoundError("Pending upload data was not found")
        if manifest.get("fingerprint") != self._fingerprint(dataset_dir):
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError("Pending upload changed after inspection. Upload the ZIP again.")
        return manifest, dataset_dir, stage_dir

    def consume(self, token: str, owner_id: str) -> tuple[dict[str, Any], Path, Path]:
        manifest, _dataset_dir, stage_dir = self.peek(token, owner_id)
        consuming_dir = contained_path(stage_dir.parent, f".consuming-{token}")
        try:
            stage_dir.replace(consuming_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError("Pending upload was already used") from exc
        return manifest, contained_path(consuming_dir, "dataset"), consuming_dir

    def consuming_path(self, token: str, owner_id: str) -> Path:
        self._validate_token(token)
        roots = self._roots(owner_id)
        for name in (f".consuming-{token}", token):
            for root in roots:
                candidate = contained_path(root, name)
                manifest = self._read_manifest(candidate)
                if manifest.get("token") == token and manifest.get("owner_id") == owner_id:
                    return contained_path(root, f".consuming-{token}")
        return contained_path(roots[0], f".consuming-{token}")

    @staticmethod
    def _read_manifest(stage_dir: Path) -> dict[str, Any]:
        try:
            data = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _fingerprint(dataset_dir: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted((item for item in dataset_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(dataset_dir).as_posix()):
            stat = path.stat()
            digest.update(path.relative_to(dataset_dir).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()
