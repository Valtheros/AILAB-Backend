"""Linux project-quota primitives for a trusted, separately privileged operator."""

from __future__ import annotations

import ctypes
import fcntl
import os
import stat
import struct
from pathlib import Path


_PRJQUOTA = 2
_Q_GETINFO = 0x800005
_Q_GETQUOTA = 0x800007
_Q_SETQUOTA = 0x800008
_QIF_BLIMITS = 1
_FS_IOC_FSGETXATTR = 0x801C581F
_FS_IOC_FSSETXATTR = 0x401C5820
_FS_XFLAG_PROJINHERIT = 0x00000200
_FSXATTR = struct.Struct('=IIIII8s')


class _QuotaInfo(ctypes.Structure):
    _fields_ = [('block_grace', ctypes.c_uint64), ('inode_grace', ctypes.c_uint64),
                ('flags', ctypes.c_uint32), ('valid', ctypes.c_uint32)]


class _QuotaBlock(ctypes.Structure):
    _fields_ = [('hard', ctypes.c_uint64), ('soft', ctypes.c_uint64), ('used', ctypes.c_uint64),
                ('inode_hard', ctypes.c_uint64), ('inode_soft', ctypes.c_uint64), ('inodes_used', ctypes.c_uint64),
                ('block_time', ctypes.c_uint64), ('inode_time', ctypes.c_uint64), ('valid', ctypes.c_uint32)]


def _project_id(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 4294967294:
        raise ValueError('Invalid filesystem project ID')
    return value


def _quotactl(device: Path, command: int, project_id: int, result: ctypes.Structure) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    operation = ctypes.c_int((command << 8) | _PRJQUOTA)
    if libc.quotactl(operation, os.fsencode(device), project_id, ctypes.byref(result)):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def project_quota_active(device: str | Path) -> bool:
    _quotactl(Path(device), _Q_GETINFO, 0, _QuotaInfo())
    return True


def _open_directory(path: str | Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)


def _attributes(fd: int) -> list:
    buffer = bytearray(_FSXATTR.size)
    fcntl.ioctl(fd, _FS_IOC_FSGETXATTR, buffer, True)
    return list(_FSXATTR.unpack(buffer))


def directory_project(path: str | Path) -> tuple[int, bool]:
    fd = _open_directory(path)
    try:
        attributes = _attributes(fd)
        return attributes[3], bool(attributes[0] & _FS_XFLAG_PROJINHERIT)
    finally:
        os.close(fd)


def assign_empty_directory(path: str | Path, project_id: int) -> None:
    """Tag a private, empty directory before any writer can access it."""
    project_id = _project_id(project_id)
    fd = _open_directory(path)
    try:
        attributes = _attributes(fd)
        if attributes[3] == project_id and attributes[0] & _FS_XFLAG_PROJINHERIT:
            return
        if attributes[3] != 0 or os.listdir(fd):
            raise ValueError('Existing or nonempty directories require an offline, verified backfill')
        attributes[0] |= _FS_XFLAG_PROJINHERIT
        attributes[3] = project_id
        fcntl.ioctl(fd, _FS_IOC_FSSETXATTR, bytearray(_FSXATTR.pack(*attributes)), True)
        changed = _attributes(fd)
        if changed[3] != project_id or not changed[0] & _FS_XFLAG_PROJINHERIT:
            raise RuntimeError('Filesystem did not retain project inheritance')
    finally:
        os.close(fd)


def _validate_device(device: str | Path, directory: str | Path, project_id: int) -> Path:
    _project_id(project_id)
    device = Path(device)
    device_stat = device.stat()
    fd = _open_directory(directory)
    try:
        attributes = _attributes(fd)
        if not stat.S_ISBLK(device_stat.st_mode) or device_stat.st_rdev != os.fstat(fd).st_dev:
            raise ValueError('Quota device does not match the owner directory')
        if attributes[3] != project_id or not attributes[0] & _FS_XFLAG_PROJINHERIT:
            raise ValueError('Directory is not assigned to the requested project')
    finally:
        os.close(fd)
    return device


def project_usage(device: str | Path, directory: str | Path, project_id: int) -> tuple[int, int]:
    device = _validate_device(device, directory, project_id)
    block = _QuotaBlock()
    _quotactl(device, _Q_GETQUOTA, project_id, block)
    return int(block.used), int(block.hard) * 1024


class ProjectQuotaMeasurement:
    """Read quota usage through trusted project anchors on one filesystem."""

    def __init__(self, device: str | Path, anchors: str | Path, filesystem_root: str | Path):
        self.device = Path(device)
        self.anchors = Path(anchors)
        self.filesystem_root = Path(filesystem_root)

    def __call__(self, project_id: int) -> tuple[int, int, int]:
        project_id = _project_id(project_id)
        root_fd = _open_directory(self.filesystem_root)
        anchors_fd = None
        anchor_fd = None
        try:
            anchors_fd = _open_directory(self.anchors)
            anchor_fd = os.open(str(project_id), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=anchors_fd)
            root_attributes = _attributes(root_fd)
            anchor_attributes = _attributes(anchor_fd)
            device_stat = self.device.stat()
            if (not stat.S_ISBLK(device_stat.st_mode)
                    or device_stat.st_rdev != os.fstat(root_fd).st_dev
                    or device_stat.st_rdev != os.fstat(anchor_fd).st_dev):
                raise ValueError('Quota anchors and filesystem root must be on the configured device')
            if root_attributes[3] != 0 or root_attributes[0] & _FS_XFLAG_PROJINHERIT:
                raise ValueError('Filesystem-wide free space requires an unrestricted root directory')
            if anchor_attributes[3] != project_id or not anchor_attributes[0] & _FS_XFLAG_PROJINHERIT:
                raise ValueError('Quota anchor is not assigned to the requested project')
            block = _QuotaBlock()
            _quotactl(self.device, _Q_GETQUOTA, project_id, block)
            filesystem = os.fstatvfs(root_fd)
            return int(block.used), int(block.hard) * 1024, filesystem.f_bavail * filesystem.f_frsize
        finally:
            if anchor_fd is not None:
                os.close(anchor_fd)
            if anchors_fd is not None:
                os.close(anchors_fd)
            os.close(root_fd)


def set_project_limit(device: str | Path, directory: str | Path, project_id: int, limit_bytes: int) -> None:
    if type(limit_bytes) is not int or not 1024 <= limit_bytes <= 2**63 - 1 or limit_bytes % 1024:
        raise ValueError('Quota limit must be a positive multiple of 1024 bytes')
    device = _validate_device(device, directory, project_id)
    # The Linux quota interface represents limits in 1024-byte blocks.
    block = _QuotaBlock(hard=limit_bytes // 1024, valid=_QIF_BLIMITS)
    _quotactl(device, _Q_SETQUOTA, project_id, block)
    if project_usage(device, directory, project_id)[1] != limit_bytes:
        raise RuntimeError('Filesystem did not retain the requested project limit')
