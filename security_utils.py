from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
import uuid
import zipfile
from pathlib import Path, PurePosixPath


MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
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
    try:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(descriptor, str(os.getpid()).encode("utf-8"))
        except FileExistsError as exc:
            raise FileExistsError(f"{label.title()} '{name}' is already being written.") from exc
        yield lock_path
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)


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
    backup_dir = target_dir.with_name(f".backup-{target_dir.name}-{uuid.uuid4().hex}")
    had_target = target_dir.exists()
    try:
        if had_target:
            target_dir.replace(backup_dir)
        staging_dir.replace(target_dir)
        if had_target:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        if target_dir.exists() and target_dir != staging_dir:
            shutil.rmtree(target_dir, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.replace(target_dir)
        raise
