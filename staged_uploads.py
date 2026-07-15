from __future__ import annotations

import json
import hashlib
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from security_utils import contained_path


STAGED_UPLOAD_TTL_SECONDS = 30 * 60
MAX_STAGED_UPLOADS_PER_USER = 3


class StagedUploadStore:
    def __init__(self, dataset_root: Path):
        self.root = contained_path(dataset_root, ".staged")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def cleanup(self) -> None:
        with self._lock:
            now = time.time()
            for item in self.root.iterdir():
                if not item.is_dir():
                    continue
                manifest = self._read_manifest(item)
                if not manifest or float(manifest.get("expires_at", 0)) <= now:
                    shutil.rmtree(item, ignore_errors=True)

    def create(self, owner_id: str, dataset_name: str, extracted_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.cleanup()
            owned = [
                item for item in self.root.iterdir()
                if item.is_dir() and self._read_manifest(item).get("owner_id") == owner_id
            ]
            if len(owned) >= MAX_STAGED_UPLOADS_PER_USER:
                raise ValueError(f"At most {MAX_STAGED_UPLOADS_PER_USER} pending dataset uploads are allowed per user")
            token = secrets.token_urlsafe(32)
            stage_dir = contained_path(self.root, token)
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
        if not token or len(token) > 128 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in token):
            raise ValueError("Invalid upload token")
        stage_dir = contained_path(self.root, token)
        manifest = self._read_manifest(stage_dir)
        if not manifest or manifest.get("token") != token:
            raise FileNotFoundError("Pending upload was not found")
        if manifest.get("owner_id") != owner_id:
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
        consuming_dir = contained_path(self.root, f".consuming-{token}")
        try:
            stage_dir.replace(consuming_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError("Pending upload was already used") from exc
        return manifest, contained_path(consuming_dir, "dataset"), consuming_dir

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
