import json
from pathlib import Path
import tempfile
import unittest
from monitoring.downloads import snapshot


class DownloadTests(unittest.TestCase):
    def test_ranges_and_assembled_prefix_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'download-manifest.json').write_text(json.dumps({'sha': 'rev', 'siblings': [
                {'rfilename': 'model.safetensors', 'size': 100, 'lfs': {'sha256': 'hash'}}]}))
            path = root / 'model.safetensors'
            path.write_bytes(b'x' * 60)
            parts = root / '.ranges'
            parts.mkdir()
            (parts / 'model.safetensors.40-79').write_bytes(b'x' * 40)
            (parts / 'model.safetensors.80-99').write_bytes(b'x' * 10)
            self.assertEqual(snapshot(root)['downloaded'], 90)
            self.assertFalse(snapshot(root)['verified'])
            path.write_bytes(b'x' * 100)
            self.assertEqual(snapshot(root)['downloaded'], 100)
            (root / 'verification.json').write_text(json.dumps({'revision': 'rev', 'files': {
                path.name: {'size': 100, 'mtime_ns': path.stat().st_mtime_ns}}}))
            self.assertTrue(snapshot(root)['verified'])
            path.write_bytes(b'x' * 50)
            self.assertFalse(snapshot(root)['verified'])
