"""Resume matching model weights with bounded HTTP ranges; validate final SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
import urllib.request

from download_parallel import ROOT, REVISION, digest

CHUNK = 64 * 1024 * 1024


def fetch(task):
    url, start, end, total, part = task
    expected = end - start + 1
    for attempt in range(12):
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            return part
        if have > expected:
            raise ValueError(f'Oversized partial file: {part}')
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start+have}-{end}'})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=45) as response:
                wanted = f'bytes {start+have}-{end}/{total}'
                if response.status != 206 or response.headers.get('Content-Range') != wanted:
                    raise ValueError(f'Server did not honor range: {response.status}')
                with part.open('ab') as out:
                    remaining = expected - have
                    while remaining:
                        data = response.read(min(1024 * 1024, remaining))
                        if not data:
                            raise OSError('Truncated range response')
                        out.write(data)
                        remaining -= len(data)
            return part
        except ValueError:
            raise
        except Exception as exc:
            print(f'Retry {part.name}: {type(exc).__name__}', flush=True)
            if attempt == 11:
                raise
            time.sleep(min(20, attempt + 2))


def main():
    manifest = json.loads((ROOT / 'download-manifest.json').read_text())
    if manifest['sha'] != REVISION:
        raise ValueError('Unexpected Hugging Face revision')
    mirror = json.loads(Path('configs/models/qwen3-8b-mirror.json').read_text(encoding='utf-8'))
    mirrors = {e['Name']: e for e in mirror['Data']['Files']}
    parts = ROOT / '.ranges'
    parts.mkdir(exist_ok=True)
    plan_path = parts / 'plan.json'
    weights = [e for e in manifest['siblings'] if e['rfilename'].endswith('.safetensors')]
    for entry in weights:
        m = mirrors[entry['rfilename']]
        if m['Size'] != entry['size'] or m['Sha256'] != entry['lfs']['sha256']:
            raise ValueError('Mirror identity mismatch')
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = {e['rfilename']: (ROOT / e['rfilename']).stat().st_size
                if (ROOT / e['rfilename']).exists() else 0 for e in weights}
        plan_path.write_text(json.dumps(plan))
    tasks = []
    for e in weights:
        name, total = e['rfilename'], e['size']
        m = mirrors[name]
        url = f"https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/{m['Revision']}/{name}"
        if plan[name] > total:
            raise ValueError('Local prefix larger than expected file')
        for start in range(plan[name], total, CHUNK):
            end = min(total, start + CHUNK) - 1
            tasks.append((url, start, end, total, parts / f'{name}.{start}-{end}'))
    print(f'Fetching {len(tasks)} ranges with 16 connections', flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for count, future in enumerate(as_completed([pool.submit(fetch, t) for t in tasks]), 1):
            future.result()
            print(f'Range {count}/{len(tasks)} complete; elapsed {time.monotonic()-started:.0f}s', flush=True)
    for e in weights:
        name, total = e['rfilename'], e['size']
        path = ROOT / name
        for start in range(plan[name], total, CHUNK):
            end = min(total, start + CHUNK) - 1
            current = path.stat().st_size if path.exists() else 0
            if current > total or current < start:
                raise ValueError('Unexpected local prefix during assembly')
            if current > end:
                continue
            with (parts / f'{name}.{start}-{end}').open('rb') as src, path.open('ab') as out:
                src.seek(current - start)
                shutil.copyfileobj(src, out, 1024 * 1024)
        if path.stat().st_size != total or digest(path) != e['lfs']['sha256']:
            raise ValueError(f'Integrity check failed: {name}')
        print(f'Verified {name}', flush=True)
    print('ALL_WEIGHTS_VERIFIED; run download_parallel.py to finish metadata verification', flush=True)


if __name__ == '__main__':
    main()
