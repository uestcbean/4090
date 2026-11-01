"""Resume fixed-revision model files on Windows, with bounded parallel transfers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

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
        # Start a fresh curl invocation after disconnects so -C re-reads the
        # actual partial file length (including bytes from the failed attempt).
        for attempt in range(20):
            with (ROOT / (name + '.download.log')).open('a') as log:
                transfer = subprocess.run(['curl.exe', '--proxy', 'http://127.0.0.1:10809', '-fL',
                    '-C', '-', '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '60',
                    f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{REVISION}/{name}?download=true',
                    '-o', str(path)], stderr=log)
            if transfer.returncode == 0:
                break
            if transfer.returncode not in (5, 6, 7, 18, 22, 28, 35, 52, 55, 56):
                transfer.check_returncode()
            print(f'Retrying {name}: attempt {attempt + 1}, curl exit {transfer.returncode}', flush=True)
            time.sleep(min(30, 3 * (attempt + 1)))
        else:
            transfer.check_returncode()
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
