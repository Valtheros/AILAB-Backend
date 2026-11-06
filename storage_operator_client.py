"""Small transport for trusted application processes, never a user-facing API."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket


class StorageOperatorError(RuntimeError):
    def __init__(self, code='STORAGE_UNAVAILABLE'):
        self.code = code
        super().__init__('Storage accounting is unavailable.' if code == 'STORAGE_UNAVAILABLE'
                         else 'Storage operation was refused.')


class StorageOperatorClient:
    def __init__(self, control: str | Path | None = None):
        self.control = Path(control or os.environ.get('STORAGE_CONTROL_DIR', '/storage-control'))

    def call(self, operation: str, project_id: int, owner: str, **values) -> dict:
        try:
            if {'token', 'operation', 'projectId', 'owner'} & values.keys():
                raise ValueError('Cannot replace storage request identity')
            token = (self.control / 'token').read_text(encoding='ascii')
            if len(token) != 64:
                raise StorageOperatorError()
            payload = json.dumps({'token': token, 'operation': operation, 'projectId': project_id,
                                  'owner': owner, **values}).encode('utf-8') + b'\n'
            if len(payload) > 8192:
                raise ValueError('Storage request is too large')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(120 if operation == 'retire' else 3)
                connection.connect(str(self.control / 'storage.sock'))
                connection.sendall(payload)
                with connection.makefile('rb') as incoming:
                    raw = incoming.readline(8193)
            if len(raw) > 8192 or not raw.endswith(b'\n'):
                raise StorageOperatorError()
            response = json.loads(raw)
            if not isinstance(response, dict) or response.get('ok') is not True:
                code = response.get('code') if isinstance(response, dict) else None
                if code not in {'STORAGE_CONFLICT', 'STORAGE_QUOTA_EXCEEDED'}:
                    code = 'STORAGE_UNAVAILABLE'
                raise StorageOperatorError(code)
            result = response['result']
            if not isinstance(result, dict) or any(type(result.get(key)) is not int or result[key] < 0
                                                 for key in ('usedBytes', 'hardLimitBytes', 'freeBytes')):
                raise StorageOperatorError()
            return result
        except StorageOperatorError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise StorageOperatorError() from exc
