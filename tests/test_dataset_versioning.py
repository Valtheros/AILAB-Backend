from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_versioning import encode_manifest, source_manifest, verify_source, write_manifest


class DatasetVersionManifestTests(unittest.TestCase):
    def test_manifest_uses_file_contents_and_ignores_generated_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'train').mkdir()
            (root / 'train' / 'image.jpg').write_bytes(b'image')
            (root / '.ailab_exports').mkdir()
            (root / '.ailab_exports' / 'cache').write_bytes(b'cache')
            metadata = {'classes': ['car'], 'tasks': ['object_detection']}
            first = source_manifest(root, metadata)
            digest = write_manifest(root, first)
            self.assertTrue(verify_source(root, digest))
            self.assertEqual(encode_manifest(source_manifest(root, metadata))[1], digest)
            (root / '.ailab_exports' / 'cache').write_bytes(b'new cache')
            self.assertEqual(encode_manifest(source_manifest(root, metadata))[1], digest)
            (root / 'train' / 'image.jpg').write_bytes(b'other')
            self.assertNotEqual(encode_manifest(source_manifest(root, metadata))[1], digest)
            with self.assertRaisesRegex(ValueError, 'Dataset changed'):
                verify_source(root, digest)

    def test_manifest_rejects_symbolic_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'image.jpg').write_bytes(b'image')
            (root / 'link').symlink_to(root / 'image.jpg')
            with self.assertRaises(ValueError):
                source_manifest(root, {})

    def test_manifest_includes_hidden_source_but_excludes_only_generated_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '.source').mkdir()
            (root / '.source' / 'image.png').write_bytes(b'image')
            (root / '.ailab_index').mkdir()
            (root / '.ailab_index' / 'thumbnail.jpg').write_bytes(b'cache')
            first = encode_manifest(source_manifest(root, {}))[1]
            (root / '.ailab_index' / 'thumbnail.jpg').write_bytes(b'new-cache')
            self.assertEqual(encode_manifest(source_manifest(root, {}))[1], first)
            (root / '.source' / 'image.png').write_bytes(b'changed')
            self.assertNotEqual(encode_manifest(source_manifest(root, {}))[1], first)


if __name__ == '__main__':
    unittest.main()
