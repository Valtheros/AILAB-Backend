"""Quota-bound multipart spooling after authentication, before parsing the body."""
from __future__ import annotations

from contextvars import ContextVar
import errno
import logging
import re
from tempfile import SpooledTemporaryFile

import anyio
from starlette import formparsers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from api_errors import PublicError, error_info
from security_utils import MAX_UPLOAD_BYTES
from storage_service import RequestLease, enabled


_spools = ContextVar('storage_upload_spools', default=None)
_OVERHEAD = 1024 * 1024
_UPLOAD_PATHS = re.compile(r'^/api/(?:datasets/inspect-upload|upload-dataset|runs/[^/]+/predict(?:-batch)?|v1/models/[^/]+/predict(?:-batch)?)$')


class QuotaSpool(SpooledTemporaryFile):
    def write(self, data):
        try:
            return super().write(data)
        except OSError as exc:
            if exc.errno not in {errno.EDQUOT, errno.ENOSPC}:
                raise
            error = HTTPException(409, 'There is not enough storage for this upload.')
            error.code = 'STORAGE_QUOTA_EXCEEDED'
            raise error from exc


def scoped_spool(*args, **kwargs):
    context = _spools.get()
    if context is None:
        if enabled():
            raise PublicError('STORAGE_UNAVAILABLE', 'Upload storage has not been reserved.', status=503)
        return SpooledTemporaryFile(*args, **kwargs)
    lease, files = context
    kwargs['dir'] = lease.directory
    result = QuotaSpool(*args, **kwargs)
    files.append(result)
    return result


def current_lease(owner_id):
    context = _spools.get()
    if context is None or context[0].owner_id != owner_id:
        raise PublicError('STORAGE_UNAVAILABLE', 'Upload storage has not been reserved for this account.', status=503)
    return context[0]


class StorageUploadMiddleware:
    def __init__(self, app):
        self.app = app
        formparsers.SpooledTemporaryFile = scoped_spool

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not enabled() or scope.get('method') != 'POST' or not _UPLOAD_PATHS.fullmatch(scope['path']):
            return await self.app(scope, receive, send)
        request = Request(scope)
        principal = scope.get('state', {}).get('model_api_principal')
        owner_id = principal['owner_user_id'] if principal else request.headers.get('x-user-id')
        lease, context_token, files = None, None, []
        response_started = False

        async def checked_send(message):
            nonlocal response_started
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)

        try:
            if not owner_id:
                raise PublicError('UNAUTHORIZED', 'Authentication required.', status=401)
            maximum = (1024**3 if scope['path'].endswith('predict-batch') else
                       10 * 1024**2 if scope['path'].endswith('predict') else MAX_UPLOAD_BYTES) + _OVERHEAD
            raw_length = request.headers.get('content-length')
            if raw_length is not None and (not raw_length.isascii() or not raw_length.isdigit() or len(raw_length) > 12):
                raise PublicError('INVALID_UPLOAD', 'Invalid upload length.')
            length = int(raw_length) if raw_length is not None else maximum
            if length > maximum:
                raise PublicError('UPLOAD_TOO_LARGE', 'The upload exceeds the supported size.', status=413)
            # Include directory metadata and multipart overhead; the endpoint reserves extraction separately.
            lease = RequestLease(owner_id, max(65536, length + 65536))
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(lease.open)
            scope.setdefault('state', {})['storage_upload_lease'] = lease
            context_token = _spools.set((lease, files))
            total = 0

            async def checked_receive():
                nonlocal total
                message = await receive()
                if message['type'] == 'http.request':
                    total += len(message.get('body', b''))
                    if total > length:
                        # FastAPI preserves HTTPException during multipart parsing, not arbitrary exceptions.
                        raise HTTPException(413, 'The upload exceeds its declared or supported size.')
                return message

            await self.app(scope, checked_receive, checked_send)
        except (PublicError, OSError) as exc:
            if isinstance(exc, OSError):
                if exc.errno not in {errno.EDQUOT, errno.ENOSPC}:
                    raise
                exc = PublicError('STORAGE_QUOTA_EXCEEDED', 'There is not enough storage for this upload.', status=409)
            if response_started:
                raise
            await JSONResponse({'detail': str(exc), **error_info(exc),
                                'requestId': scope.get('state', {}).get('request_id')}, status_code=exc.status)(scope, receive, send)
        finally:
            if context_token is not None:
                _spools.reset(context_token)
            if lease is not None:
                # Close spools before returning their allowance, also on disconnect/cancellation.
                with anyio.CancelScope(shield=True):
                    for file in files:
                        try:
                            file.close()
                        except OSError:
                            logging.getLogger(__name__).exception('Could not close upload spool')
                    await anyio.to_thread.run_sync(lease.close)
