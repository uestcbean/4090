"""Protocol tests use fixtures, not model performance evidence."""
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from monitoring.server import Handler


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, payload, stream):
        # Only mock the upstream engine call; the local HTTP request is real.
        real_open = urllib.request.urlopen
        def route(request, *args, **kwargs):
            if request.full_url.endswith('/v1/chat/completions'):
                return io.BytesIO(stream)
            return real_open(request, *args, **kwargs)
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.server.server_port}/api/generate',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with patch('monitoring.server.urllib.request.urlopen', side_effect=route):
            try:
                response = real_open(request, timeout=5)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                return response.code, json.load(response)

    def test_complete_stream(self):
        stream = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        stream += b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        stream += b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        code, result = self.post({'prompt': 'test'}, stream)
        self.assertEqual(code, 200)
        self.assertEqual(result['content'], 'hello')
        self.assertEqual(result['usage']['completion_tokens'], 1)
        self.assertGreaterEqual(result['elapsed_seconds'], result['first_content_seconds'])

    def test_truncated_stream_fails(self):
        code, result = self.post({'prompt': 'test'}, b'data: {"choices":[{"delta":{"content":"partial"}}]}\n')
        self.assertEqual(code, 502)
        self.assertEqual(result['status'], 'failed')

    def test_invalid_payload(self):
        for payload in ([], {}, {'prompt': ''}, {'prompt': 'x' * 8001}):
            with self.subTest(payload_type=type(payload).__name__):
                code, result = self.post(payload, b'')
                self.assertEqual(code, 400)
                self.assertEqual(result['status'], 'failed')
