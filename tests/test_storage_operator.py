import json
import os
from pathlib import Path
import socket
import socketserver
import tempfile
import threading
import unittest
from unittest.mock import Mock

from storage_operator import OperatorHandler, open_relative, owner_digest, relative_parts
from storage_operator_client import StorageOperatorClient, StorageOperatorError


class StoragePathTests(unittest.TestCase):
    def test_invalid_paths_never_open_a_directory(self):
        for path in ('', '/', '../other', 'a/../b', '/absolute', 'a//b', 'a/./b', 'a\\b', 'a\x00b', None):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_parts(path)

    def test_symlink_ancestors_cannot_escape_or_create_outside_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            (root / 'outside').mkdir()
            (root / 'data' / 'link').symlink_to(root / 'outside', target_is_directory=True)
            fd = os.open(root / 'data', os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaises(OSError):
                    open_relative(fd, 'link/created', create=True)
                self.assertFalse((root / 'outside' / 'created').exists())
                child = open_relative(fd, 'new/child', create=True)
                self.assertTrue((root / 'data' / 'new' / 'child').is_dir())
                os.close(child)
            finally:
                os.close(fd)

    def test_owner_binding_uses_full_digest_and_rejects_missing_owners(self):
        self.assertEqual(len(owner_digest('alice')), 64)
        self.assertNotEqual(owner_digest('alice'), owner_digest('bob'))
        for owner in ('', None, 'x' * 1025):
            with self.assertRaises(ValueError):
                owner_digest(owner)


class StorageTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.control = Path(self.temporary.name)
        (self.control / 'token').write_text('a' * 64)
        self.server = socketserver.UnixStreamServer(str(self.control / 'storage.sock'), OperatorHandler)
        self.server.token = 'a' * 64
        self.server.operator = Mock()
        self.server.operator.dispatch.return_value = {'usedBytes': 4096, 'hardLimitBytes': 8192, 'freeBytes': 10000}
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.client = StorageOperatorClient(self.control)

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def test_authenticated_rpc_never_passes_token_to_operator(self):
        self.assertEqual(self.client.call('measure', 1000000, 'alice')['usedBytes'], 4096)
        self.server.operator.dispatch.assert_called_once_with(
            {'operation': 'measure', 'projectId': 1000000, 'owner': 'alice'})

    def test_wrong_token_cannot_reach_privileged_operation(self):
        (self.control / 'token').write_text('b' * 64)
        with self.assertRaises(StorageOperatorError):
            self.client.call('limit', 1000000, 'alice', limitBytes=1024)
        self.server.operator.dispatch.assert_not_called()

    def test_oversized_and_non_object_messages_are_rejected(self):
        for payload in (b'x' * 8193 + b'\n', b'[]\n'):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.control / 'storage.sock'))
                connection.sendall(payload)
                response = json.loads(connection.recv(8192))
                self.assertFalse(response['ok'])
        self.server.operator.dispatch.assert_not_called()

    def test_invalid_measurements_and_missing_socket_fail_closed(self):
        for result in ({}, {'usedBytes': True, 'hardLimitBytes': 1, 'freeBytes': 1},
                       {'usedBytes': -1, 'hardLimitBytes': 1, 'freeBytes': 1}):
            self.server.operator.dispatch.return_value = result
            with self.assertRaises(StorageOperatorError):
                self.client.call('measure', 1000000, 'alice')
        with self.assertRaises(StorageOperatorError):
            StorageOperatorClient(self.control / 'missing').call('measure', 1000000, 'alice')


if __name__ == '__main__':
    unittest.main()
