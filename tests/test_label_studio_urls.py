import os
import unittest
from unittest.mock import patch

from label_studio_client import BridgeError, public_url


class PublicUrlTests(unittest.TestCase):
    def test_subpath_is_kept_and_missing_or_invalid_urls_fail(self):
        url = 'https://ai-lab.cs.psu.ac.th/label-studio'
        with patch.dict(os.environ, {'LABEL_STUDIO_PUBLIC_URL': url + '/'}):
            self.assertEqual(public_url(), url)
        for value in ('', '/label-studio', 'ftp://example.com', 'https://user:secret@example.com',
                      'https://example.com/?next=other', 'https://example.com/#fragment'):
            with self.subTest(value=value), patch.dict(os.environ, {'LABEL_STUDIO_PUBLIC_URL': value}):
                with self.assertRaises(BridgeError):
                    public_url()
