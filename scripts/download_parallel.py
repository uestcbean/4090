"""Resume fixed-revision model files on Windows, with bounded parallel transfers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
ROOT = Path('D:/ai/models/Qwen3-8B') / REVISION


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(entry):
    name = entry['rfilename']
    if Path(name).name != name or '\\' in name:
        raise ValueError('Unexpected filename')
    path = ROOT / name
    checksum = entry.get('lfs', {}).get('sha256')
    if not path.exists() or path.stat().st_size != entry['size']:
        print('Downloading', name, flush=True)
        with (ROOT / (name + '.download.log')).open('a') as log:
            subprocess.run(['curl.exe', '--proxy', 'http://127.0.0.1:10809', '-fL',
                '-C', '-', '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30',
                f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{REVISION}/{name}?download=true',
                '-o', str(path)], stderr=log, check=True)
    if path.stat().st_size != entry['size']:
        raise ValueError(f'Size mismatch: {name}')
    if checksum and digest(path) != checksum:
        raise ValueError(f'SHA256 mismatch: {name}')
    print('Verified', name, flush=True)


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'download-manifest.json').read_text())
    assert manifest['sha'] == REVISION
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(download, [e for e in manifest['siblings'] if e['rfilename'] != '.gitattributes']))
    print('MODEL_DOWNLOAD_VERIFIED', flush=True)
