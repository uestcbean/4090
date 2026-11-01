"""Report unique downloaded bytes, excluding duplicated assembled range parts."""
import json
from pathlib import Path
import re
import time

MODEL_ROOT = Path('D:/ai/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218')


def snapshot(root=MODEL_ROOT):
    manifest = json.loads((root / 'download-manifest.json').read_text(encoding='utf-8'))
    receipt_path = root / 'verification.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    trusted = receipt.get('revision') == manifest['sha']
    rows = []
    for entry in manifest['siblings']:
        name, total = entry['rfilename'], entry['size']
        if name == '.gitattributes':
            continue
        path = root / name
        stat = path.stat() if path.exists() else None
        assembled = min(stat.st_size, total) if stat else 0
        intervals = [(0, assembled)]
        for part in (root / '.ranges').glob(name + '.*'):
            match = re.fullmatch(re.escape(name) + r'\.(\d+)-(\d+)', part.name)
            if match:
                start, end = map(int, match.groups())
                intervals.append((start, min(total, end + 1, start + part.stat().st_size)))
        downloaded, cursor = 0, 0
        for start, end in sorted(intervals):
            downloaded += max(0, end - max(start, cursor))
            cursor = max(cursor, end)
        evidence = receipt.get('files', {}).get(name, {})
        verified = bool(trusted and stat and stat.st_size == total and
                        evidence.get('size') == total and evidence.get('mtime_ns') == stat.st_mtime_ns)
        rows.append({'name': name, 'total': total, 'downloaded': downloaded,
                     'verified': verified, 'check': 'SHA256' if entry.get('lfs') else '文件大小'})
    return {'time': time.time(), 'model': 'Qwen3-8B', 'path': str(root),
            'source': 'ModelScope 权重 / Hugging Face 配套文件', 'files': rows,
            'total': sum(r['total'] for r in rows), 'downloaded': sum(r['downloaded'] for r in rows),
            'verified': all(r['verified'] for r in rows), 'verified_at': receipt.get('verified_at')}
