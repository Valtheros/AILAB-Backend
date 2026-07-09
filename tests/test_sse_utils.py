from __future__ import annotations

import unittest

from sse_utils import heartbeat, sse_event


class SSEUtilsTests(unittest.TestCase):
    def test_formats_json_event(self):
        self.assertEqual(sse_event("status", {"status": "running"}), 'event: status\ndata: {"status":"running"}\n\n')

    def test_formats_heartbeat_comment(self):
        self.assertEqual(heartbeat(), ": heartbeat\n\n")


if __name__ == "__main__":
    unittest.main()
