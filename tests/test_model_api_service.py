from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_errors import PublicError
import model_api_service as service


class ModelAPIServiceTests(unittest.TestCase):
    def test_public_api_is_fail_closed_until_storage_enforcement_is_complete(self):
        with patch.dict(os.environ, {'MODEL_API_ENABLED': 'true'}), \
             patch.object(service, 'STORAGE_ENFORCEMENT_IMPLEMENTED', False):
            self.assertFalse(service.enabled())
            with self.assertRaises(PublicError) as error:
                service.authenticate('Bearer invalid')
            self.assertEqual(error.exception.status, 404)

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
