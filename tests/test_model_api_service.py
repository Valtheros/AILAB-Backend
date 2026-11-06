from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_errors import PublicError
import model_api_service as service


class ModelAPIServiceTests(unittest.TestCase):
    def test_public_api_requires_storage_enforcement(self):
        with patch.dict(os.environ, {'MODEL_API_ENABLED': 'true', 'STORAGE_ENFORCEMENT': 'false'}):
            self.assertFalse(service.enabled())
            with self.assertRaises(PublicError) as error:
                service.authenticate('Bearer invalid')
            self.assertEqual(error.exception.status, 404)

    def test_public_api_requires_both_deployment_flags(self):
        for storage, api in [('false', 'false'), ('true', 'false'), ('false', 'true'), ('true', 'true')]:
            with self.subTest(storage=storage, api=api), patch.dict(os.environ, {
                'MODEL_API_ENABLED': api, 'STORAGE_ENFORCEMENT': storage,
            }):
                self.assertEqual(service.enabled(), storage == api == 'true')

    def test_checkpoint_path_and_content_are_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            run.mkdir()
            (run / 'best.pt').write_bytes(b'original')
            with patch.object(service, 'RUNS_DIR', root):
                path, digest = service.checkpoint('run', 'resnet')
                service.verify_checkpoint('run', 'resnet', path, digest)
                (run / 'best.pt').write_bytes(b'changed')
                with self.assertRaises(PublicError) as error:
                    service.verify_checkpoint('run', 'resnet', path, digest)
                self.assertEqual(error.exception.code, 'CHECKPOINT_CHANGED')
