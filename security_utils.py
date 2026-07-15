from __future__ import annotations

import os
import fcntl
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
import uuid
import zipfile
from pathlib import Path, PurePosixPath


MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = max(10_000, int(os.getenv("AILAB_MAX_ARCHIVE_ENTRIES", "250000")))
MAX_ARCHIVE_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def validate_slug(value: str, label: str = "identifier") -> str:
    base_name = value.split(".", 1)[0].lower()
    if value in {".", ".."} or value.endswith(".") or base_name in WINDOWS_RESERVED_NAMES or not SLUG_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid {label}. Use 1-128 letters, numbers, dots, underscores, or hyphens; "
            "the first character must be a letter or number."
        )
    return value


def contained_path(root: Path, *parts: str | Path) -> Path:
    root_path = root.resolve()
    candidate = root_path.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root_path):
        raise ValueError(f"Path escapes configured root: {candidate}")
    return candidate


@contextmanager
def named_file_lock(root: Path, name: str, label: str = "resource"):
    validate_slug(name, label)
    locks_dir = contained_path(root, ".locks")
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = contained_path(locks_dir, f"{name}.lock")
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"{label.title()} '{name}' is already being written.") from exc
        acquired = True
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode("utf-8"))
        yield lock_path
    finally:
        if descriptor is not None:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


class ReversibleDirectoryReplace:
    def __init__(self, staging_dir: Path, target_dir: Path):
        self.staging_dir = staging_dir
        self.target_dir = target_dir
        self.backup_dir = target_dir.with_name(f".backup-{target_dir.name}-{uuid.uuid4().hex}")
        self.had_target = target_dir.exists()
        self.applied = False
        self.finished = False

    def apply(self) -> "ReversibleDirectoryReplace":
        if self.had_target:
            self.target_dir.replace(self.backup_dir)
        try:
            self.staging_dir.replace(self.target_dir)
        except Exception:
            if self.backup_dir.exists():
                self.backup_dir.replace(self.target_dir)
            raise
        self.applied = True
        return self

    def commit(self) -> None:
        if self.finished:
            return
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)
        self.finished = True

    def rollback(self) -> None:
        if self.finished:
            return
        if self.applied and self.target_dir.exists():
            self.staging_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.staging_dir.exists():
                shutil.rmtree(self.staging_dir)
            self.target_dir.replace(self.staging_dir)
        if self.backup_dir.exists():
            self.backup_dir.replace(self.target_dir)
        self.finished = True


class ReversibleDirectoryRemoval:
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.removed_dir = target_dir.with_name(f".deleting-{target_dir.name}-{uuid.uuid4().hex}")
        self.applied = False
        self.finished = False

    def apply(self) -> "ReversibleDirectoryRemoval":
        self.target_dir.replace(self.removed_dir)
        self.applied = True
        return self

    def commit(self) -> None:
        if self.finished:
            return
        if self.removed_dir.exists():
            shutil.rmtree(self.removed_dir, ignore_errors=True)
        self.finished = True

    def rollback(self) -> None:
        if self.finished:
            return
        if self.applied and self.removed_dir.exists() and not self.target_dir.exists():
            self.removed_dir.replace(self.target_dir)
        self.finished = True


def _validate_archive_member(member: zipfile.ZipInfo, target_dir: Path) -> None:
    filename = member.filename.replace("\\", "/")
    if not filename or filename.startswith("/") or re.match(r"^[A-Za-z]:", filename):
        raise ValueError("ZIP contains an unsafe absolute path")
    contained_path(target_dir, filename)
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError("ZIP symbolic links are not supported")
    if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
        raise ValueError("ZIP entry exceeds the maximum allowed size")
    if member.file_size and member.compress_size == 0:
        raise ValueError("ZIP entry has an unsafe compression ratio")
    if member.compress_size and member.file_size / member.compress_size > MAX_COMPRESSION_RATIO:
        raise ValueError("ZIP entry has an unsafe compression ratio")


def safe_extract_zip(zip_file: zipfile.ZipFile, target_dir: Path) -> None:
    members = zip_file.infolist()
    if len(members) > MAX_ARCHIVE_ENTRIES:
        raise ValueError("ZIP contains too many entries")

    normalized_names = [member.filename.replace("\\", "/").rstrip("/") for member in members]
    implied_directories = {
        str(parent)
        for name in normalized_names
        for parent in PurePosixPath(name).parents
        if str(parent) not in {"", "."}
    }

    total_uncompressed = 0
    for member in members:
        _validate_archive_member(member, target_dir)
        total_uncompressed += member.file_size
        if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP expands beyond the maximum allowed size")

    target_dir.mkdir(parents=True, exist_ok=False)
    for member in members:
        filename = member.filename.replace("\\", "/").rstrip("/")
        destination = contained_path(target_dir, filename)
        is_implied_directory = filename in implied_directories
        if member.is_dir() or (is_implied_directory and member.file_size == 0):
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if is_implied_directory:
            raise ValueError("ZIP contains a file and directory with the same path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(member, "r") as source, open(destination, "wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


async def save_upload_to_temp(file, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".upload-", suffix=".zip", dir=root)
    temp_path = Path(filename)
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError("ZIP upload exceeds the maximum allowed size")
                output.write(chunk)
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def staging_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".dataset-", dir=root))


def replace_directory(staging_dir: Path, target_dir: Path) -> None:
    operation = ReversibleDirectoryReplace(staging_dir, target_dir)
    try:
        operation.apply()
        operation.commit()
    except Exception:
        operation.rollback()
        raise
