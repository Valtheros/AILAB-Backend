"""Private Unix-socket quota operator; no shell commands or public network listener."""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import socketserver
import stat

from storage_project_quota import (
    ProjectQuotaMeasurement, _attributes, _FSXATTR, _FS_IOC_FSSETXATTR,
    _FS_XFLAG_PROJINHERIT, _project_id, project_quota_active, set_project_limit,
)


MAX_REQUEST_BYTES = 8192
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NAMESPACES = {'dataset': 0, 'runs': 0, 'storage': 0, 'label-studio': 1001}


def relative_parts(value: str) -> tuple[str, ...]:
    if (not isinstance(value, str) or not value or len(value) > 2048
            or '\x00' in value or '\\' in value or value.startswith('/')):
        raise ValueError('Invalid storage-relative path')
    parts = value.split('/')
    if len(parts) > 12 or any(part in {'', '.', '..'} for part in parts):
        raise ValueError('Invalid storage-relative path')
    return tuple(parts)


def open_relative(root_fd: int, value: str, *, create=False, project_id=None) -> int:
    """Keep every ancestor pinned; never follow a writable parent's symlink."""
    parts = relative_parts(value)
    device = os.fstat(root_fd).st_dev
    current = os.dup(root_fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
            if os.fstat(current).st_dev != device:
                raise ValueError('Storage path crosses a filesystem boundary')
            if project_id is not None and _attributes(current)[3] not in {0, project_id}:
                raise ValueError('Storage path belongs to a different owner')
        result, current = current, None
        return result
    finally:
        if current is not None:
            os.close(current)


def assign_empty_fd(fd: int, project_id: int) -> None:
    _project_id(project_id)
    attributes = _attributes(fd)
    if attributes[3] == project_id and attributes[0] & _FS_XFLAG_PROJINHERIT:
        return
    if attributes[3] != 0 or os.listdir(fd):
        raise ValueError('Existing files require a verified owner backfill')
    attributes[0] |= _FS_XFLAG_PROJINHERIT
    attributes[3] = project_id
    fcntl.ioctl(fd, _FS_IOC_FSSETXATTR, bytearray(_FSXATTR.pack(*attributes)), True)
    changed = _attributes(fd)
    if changed[3] != project_id or not changed[0] & _FS_XFLAG_PROJINHERIT:
        raise RuntimeError('Filesystem did not retain project inheritance')


def owner_digest(owner: str) -> str:
    if not isinstance(owner, str) or not owner or len(owner.encode('utf-8')) > 1024:
        raise ValueError('Invalid storage owner')
    return hashlib.sha256(owner.encode('utf-8')).hexdigest()


def clear_directory(fd: int, project_id: int):
    for name in os.listdir(fd):
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if info.st_dev != os.fstat(fd).st_dev:
            raise ValueError('Temporary files cross a filesystem boundary')
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=fd)
            try:
                if _attributes(child)[3] != project_id:
                    raise ValueError('Temporary directory belongs to a different owner')
                clear_directory(child, project_id)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=fd)
        else:
            if not stat.S_ISLNK(info.st_mode):
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('Unexpected special file in user storage')
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                try:
                    if not stat.S_ISREG(info.st_mode) or _attributes(child)[3] != project_id:
                        raise ValueError('File belongs to a different owner or is not regular')
                finally:
                    os.close(child)
            os.unlink(name, dir_fd=fd)


def remove_owned(fd: int, project_id: int):
    """Account removal follows kernel ownership, not user-supplied file paths."""
    device = os.fstat(fd).st_dev
    for name in os.listdir(fd):
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if info.st_dev != device:
            raise ValueError('Storage crosses a filesystem boundary')
        if stat.S_ISLNK(info.st_mode):
            continue
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError('Unexpected special file in user storage')
        child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
        try:
            assigned = _attributes(child)[3]
            owned = assigned == project_id
            if stat.S_ISDIR(info.st_mode):
                if owned:
                    clear_directory(child, project_id)
                elif assigned == 0:
                    remove_owned(child, project_id)
        finally:
            os.close(child)
        if owned:
            if stat.S_ISDIR(info.st_mode):
                os.rmdir(name, dir_fd=fd)
            else:
                os.unlink(name, dir_fd=fd)


class QuotaOperator:
    def __init__(self, root: Path, device: Path):
        self.root = root
        self.device = device
        self.anchors = root / '.quota'
        root_fd = os.open(root, _DIRECTORY_FLAGS)
        try:
            device_stat = device.stat()
            if (not stat.S_ISBLK(device_stat.st_mode)
                    or os.fstat(root_fd).st_dev != device_stat.st_rdev
                    or _attributes(root_fd)[3] != 0):
                raise ValueError('User-data filesystem does not match its configured device')
            project_quota_active(device)
            try:
                os.mkdir('.quota', mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            anchor_fd = open_relative(root_fd, '.quota')
            os.close(anchor_fd)
        finally:
            os.close(root_fd)
        self.measure = ProjectQuotaMeasurement(device, self.anchors, root)

    def _anchor(self, project_id: int, owner: str, *, provision=False) -> Path:
        _project_id(project_id)
        digest = owner_digest(owner)
        root_fd = os.open(self.anchors, _DIRECTORY_FLAGS)
        try:
            fd = open_relative(root_fd, str(project_id), create=provision, project_id=project_id)
        finally:
            os.close(root_fd)
        try:
            if provision:
                assign_empty_fd(fd, project_id)
            name = f'owner-{digest}'
            entries = os.listdir(fd)
            if not entries and provision:
                # mkdir makes the owner binding atomic even if the operator is killed.
                os.mkdir(name, mode=0o700, dir_fd=fd)
                os.fsync(fd)
                entries = [name]
            if entries != [name]:
                raise ValueError('Filesystem project belongs to a different owner')
            binding = os.open(name, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(binding)
        finally:
            os.close(fd)
        return self.anchors / str(project_id)

    def dispatch(self, request: dict) -> dict:
        operation = request.get('operation')
        if operation == 'capacity':
            fd = os.open(self.root, _DIRECTORY_FLAGS)
            try:
                usage = os.fstatvfs(fd)
                return {'usedBytes': 0, 'hardLimitBytes': 0, 'freeBytes': usage.f_bavail * usage.f_frsize}
            finally:
                os.close(fd)
        if operation not in {'provision', 'measure', 'limit', 'prepare', 'cleanup-upload', 'retire'}:
            raise ValueError('Unknown storage operation')
        project_id, owner = request.get('projectId'), request.get('owner')
        anchor = self._anchor(project_id, owner, provision=operation == 'provision')
        if operation == 'provision':
            used, limit, _ = self.measure(project_id)
            if limit == 0:
                set_project_limit(self.device, anchor, project_id, max(1024, ((used + 1023) // 1024) * 1024))
        elif operation == 'limit':
            set_project_limit(self.device, anchor, project_id, request.get('limitBytes'))
        elif operation == 'retire':
            root_fd = os.open(self.root, _DIRECTORY_FLAGS)
            try:
                for namespace in _NAMESPACES:
                    base = open_relative(root_fd, namespace)
                    try:
                        remove_owned(base, project_id)
                    finally:
                        os.close(base)
            finally:
                os.close(root_fd)
            # Retain only the immutable owner binding; project IDs are never reused.
            expected = anchor.stat().st_blocks * 512 + sum(p.stat().st_blocks * 512 for p in anchor.iterdir())
            used, _, _ = self.measure(project_id)
            if used != expected:
                raise ValueError('Unremoved owner data remains on the filesystem')
            set_project_limit(self.device, anchor, project_id, max(1024, ((used + 1023) // 1024) * 1024))
        elif operation == 'prepare':
            namespace = request.get('namespace')
            if namespace not in _NAMESPACES:
                raise ValueError('Unknown storage namespace')
            root_fd = os.open(self.root, _DIRECTORY_FLAGS)
            base_fd = None
            fd = None
            try:
                base_fd = open_relative(root_fd, namespace)
                fd = open_relative(base_fd, request.get('path'), create=True, project_id=project_id)
                assign_empty_fd(fd, project_id)
                os.fchown(fd, _NAMESPACES[namespace], 0)
            finally:
                if fd is not None:
                    os.close(fd)
                if base_fd is not None:
                    os.close(base_fd)
                os.close(root_fd)
        elif operation == 'cleanup-upload':
            namespace, request_id = request.get('namespace'), request.get('requestId')
            if namespace not in {'storage', 'label-studio'} or not isinstance(request_id, str) or not re.fullmatch('[0-9a-f]{32}', request_id):
                raise ValueError('Invalid temporary upload reference')
            root_fd = os.open(self.root, _DIRECTORY_FLAGS)
            parent = fd = None
            try:
                parent = open_relative(root_fd, f'{namespace}/.uploads/{owner_digest(owner)[:32]}', project_id=project_id)
                fd = open_relative(parent, request_id, project_id=project_id)
                if _attributes(fd)[3] != project_id:
                    raise ValueError('Temporary directory is not owned')
                clear_directory(fd, project_id)
                os.rmdir(request_id, dir_fd=parent)
            except FileNotFoundError:
                pass
            finally:
                if fd is not None:
                    os.close(fd)
                if parent is not None:
                    os.close(parent)
                os.close(root_fd)
        used, limit, free = self.measure(project_id)
        return {'usedBytes': used, 'hardLimitBytes': limit, 'freeBytes': free}


class OperatorHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        try:
            raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b'\n'):
                raise ValueError('Invalid storage request')
            request = json.loads(raw)
            if (not isinstance(request, dict) or not isinstance(request.get('token'), str)
                    or not hmac.compare_digest(request.pop('token'), self.server.token)):
                result = {'ok': False, 'code': 'UNAUTHORIZED'}
            else:
                result = {'ok': True, 'result': self.server.operator.dispatch(request)}
        except (ValueError, TypeError, KeyError):
            result = {'ok': False, 'code': 'STORAGE_CONFLICT'}
        except OSError as exc:
            result = {'ok': False, 'code': 'STORAGE_QUOTA_EXCEEDED' if exc.errno in {errno.EDQUOT, errno.ENOSPC} else 'STORAGE_UNAVAILABLE'}
        except Exception:
            logging.exception('Quota operator failed')
            result = {'ok': False, 'code': 'STORAGE_UNAVAILABLE'}
        try:
            self.wfile.write(json.dumps(result).encode('utf-8') + b'\n')
        except OSError:
            pass  # The caller may time out; retries must inspect the retained quota.


def serve(root: Path, device: Path, control: Path):
    control.mkdir(mode=0o755, parents=True, exist_ok=True)
    lock_fd = os.open(control / 'operator.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control_fd = os.open(control, _DIRECTORY_FLAGS)
    try:
        leases_fd = open_relative(control_fd, 'leases', create=True)
        try:
            os.fchown(leases_fd, 1001, 0)
            os.fchmod(leases_fd, 0o770)
            registry_fd = os.open('.registry.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o660, dir_fd=leases_fd)
            try:
                os.fchown(registry_fd, 1001, 0)
                os.fchmod(registry_fd, 0o660)
            finally:
                os.close(registry_fd)
        finally:
            os.close(leases_fd)
    finally:
        os.close(control_fd)
    token_path = control / 'token'
    try:
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    except FileExistsError:
        pass
    else:
        try:
            os.fchown(fd, 1001, 0)
            os.write(fd, secrets.token_hex(32).encode('ascii'))
            os.fsync(fd)
        finally:
            os.close(fd)
    fd = os.open(token_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        token = os.read(fd, 65).decode('ascii')
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or len(token) != 64:
            raise ValueError('Invalid quota control token')
    finally:
        os.close(fd)
    operator = QuotaOperator(root, device)
    socket_path = control / 'storage.sock'
    if socket_path.exists():
        if not stat.S_ISSOCK(socket_path.lstat().st_mode):
            raise ValueError('Control path is not a socket')
        socket_path.unlink()
    with socketserver.UnixStreamServer(str(socket_path), OperatorHandler) as server:
        os.chown(socket_path, 1001, 0)
        os.chmod(socket_path, 0o600)
        server.operator, server.token = operator, token
        server.serve_forever(poll_interval=0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    serve(args.root, args.device, args.control)
