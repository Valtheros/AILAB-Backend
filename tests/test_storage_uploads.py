import asyncio
import errno
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from storage_uploads import QuotaSpool, StorageUploadMiddleware, _UPLOAD_PATHS
from starlette.exceptions import HTTPException


class StorageUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.leases = []
        parent = self

        class Lease:
            def __init__(self, owner, required):
                self.owner = owner
                self.required = required
                self.directory = parent.root / owner
                self.files = []
                self.closed = False
                parent.leases.append(self)

            def open(self):
                self.directory.mkdir(exist_ok=True)
                return self

            def close(self):
                parent.assertTrue(all(file.closed for file in self.files))
                self.closed = True

        for mocked in (patch('storage_uploads.RequestLease', Lease), patch('storage_uploads.enabled', return_value=True)):
            mocked.start()
            self.addCleanup(mocked.stop)
        self.app = FastAPI()
        self.app.add_middleware(StorageUploadMiddleware)

        @self.app.middleware('http')
        async def authenticated(request, call_next):
            if request.headers.get('x-internal-token') != 'test-token':
                return JSONResponse({'detail': 'Unauthorized'}, 401)
            return await call_next(request)

        @self.app.post('/api/datasets/inspect-upload')
        async def upload(request: Request, file: UploadFile = File(...)):
            lease = request.state.storage_upload_lease
            lease.files.append(file.file)
            actual = os.readlink(f'/proc/self/fd/{file.file.fileno()}')
            self.assertTrue(actual.startswith(str(lease.directory) + '/'), actual)
            if request.headers.get('x-fail'):
                raise OSError(errno.EDQUOT, 'Disk quota exceeded')
            return {'owner': lease.owner, 'bytes': len(await file.read())}

        self.headers = {'x-user-id': 'alice', 'x-internal-token': 'test-token'}

    def test_rollover_stays_inside_owner_storage_and_closes_before_release(self):
        with TestClient(self.app) as client:
            response = client.post('/api/datasets/inspect-upload', headers=self.headers,
                                   files={'file': ('test.zip', b'x' * (2 * 1024**2))})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['bytes'], 2 * 1024**2)
        self.assertTrue(self.leases[0].closed)

    def test_authentication_runs_before_reservation_and_parsing(self):
        with TestClient(self.app) as client:
            response = client.post('/api/datasets/inspect-upload', headers={'x-user-id': 'alice'},
                                   files={'file': ('test.zip', b'test')})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.leases, [])

    def test_actual_body_cannot_exceed_a_forged_content_length(self):
        with TestClient(self.app) as client:
            response = client.post('/api/datasets/inspect-upload', headers={**self.headers, 'content-length': '1'},
                                   files={'file': ('test.zip', b'test')})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertTrue(self.leases[0].closed)

    def test_oversized_header_is_rejected_before_creating_storage(self):
        with TestClient(self.app) as client:
            response = client.post('/api/datasets/inspect-upload',
                headers={**self.headers, 'content-length': str(3 * 1024**3)}, content=b'')
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.leases, [])

    def test_quota_error_is_structured_and_spools_are_closed(self):
        with TestClient(self.app) as client:
            response = client.post('/api/datasets/inspect-upload', headers={**self.headers, 'x-fail': '1'},
                                   files={'file': ('test.zip', b'test')})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()['code'], 'STORAGE_QUOTA_EXCEEDED')
        self.assertTrue(self.leases[0].closed)

    def test_concurrent_upload_contexts_do_not_share_a_temp_directory(self):
        def upload(owner):
            with TestClient(self.app) as client:
                return client.post('/api/datasets/inspect-upload', headers={**self.headers, 'x-user-id': owner},
                                   files={'file': ('test.zip', b'test')}).json()['owner']
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(upload, ['alice', 'bob'])), ['alice', 'bob'])
        self.assertTrue(all(lease.closed for lease in self.leases))

    def test_disconnected_request_releases_spools(self):
        async def inner(scope, receive, send):
            await Request(scope, receive).body()

        messages = iter([{'type': 'http.request', 'body': b'x', 'more_body': True}, {'type': 'http.disconnect'}])
        async def receive():
            return next(messages)
        async def send(message):
            pass
        scope = {'type': 'http', 'method': 'POST', 'path': '/api/upload-dataset', 'headers': [(b'x-user-id', b'alice')], 'state': {}}
        with self.assertRaises(ClientDisconnect):
            asyncio.run(StorageUploadMiddleware(inner)(scope, receive, send))
        self.assertTrue(self.leases[0].closed)

    def test_all_current_multipart_routes_are_covered(self):
        for path in ['/api/upload-dataset', '/api/datasets/inspect-upload', '/api/runs/test/predict',
                     '/api/runs/test/predict-batch', '/api/v1/models/test/predict', '/api/v1/models/test/predict-batch']:
            self.assertIsNotNone(_UPLOAD_PATHS.fullmatch(path))

    def test_parser_disk_full_keeps_a_readable_http_error(self):
        with QuotaSpool(max_size=1) as file, patch.object(file, 'rollover', side_effect=OSError(errno.EDQUOT, 'full')):
            with self.assertRaises(HTTPException) as error:
                file.write(b'too big')
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(error.exception.code, 'STORAGE_QUOTA_EXCEEDED')


if __name__ == '__main__':
    unittest.main()
