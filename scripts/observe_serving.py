"""Small real-service demonstration; not a capacity or p95 benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitoring.server import parse_metrics, summarize


def metrics(url):
    with urllib.request.urlopen(url + '/metrics', timeout=10) as response:
        raw = response.read().decode()
    parsed = parse_metrics(raw)
    if not any(key.startswith('vllm:') for key in parsed):
        raise RuntimeError('No vLLM metrics received')
    return parsed


def request(url, prompt):
    body = {'model': 'qwen3-8b', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 128, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}}
    start = time.perf_counter()
    result = {'status': 'failed', 'first_content_seconds': None, 'usage': None}
    chunks, done, finish = [], False, None
    try:
        req = urllib.request.Request(url + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                event = line[5:].strip()
                if event == '[DONE]':
                    done = True
                    break
                event = json.loads(event)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    result['usage'] = event['usage']
                for choice in event.get('choices', []):
                    content = choice.get('delta', {}).get('content')
                    if content:
                        if not chunks:
                            result['first_content_seconds'] = time.perf_counter() - start
                        chunks.append(content)
                    finish = choice.get('finish_reason') or finish
        if not done or not chunks or not finish or not result['usage']:
            raise RuntimeError('Incomplete stream or missing usage')
        result.update(status='passed', finish_reason=finish)
    except Exception as exc:
        result['error'] = str(exc)
    result.update(elapsed_seconds=time.perf_counter() - start, content=''.join(chunks))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # A new run gets a distinct early prefix. Repeated requests within the run
    # share enough tokens to span multiple cache blocks.
    prefix = f'Experiment {time.time_ns()}.\n' + (
        'KV caching retains attention keys and values from earlier tokens. '
        'Prefill processes the prompt; decode generates the continuation.\n' * 40)
    prompt = prefix + '\n请用中文总结以上内容，解释缓存为何能避免重复计算。'
    report = {'note': 'Demonstration only; warmup, queueing and noise confound latency comparison. '
              'First-content timing is not token timing. Metrics cover all engine traffic.',
              'prompt': prompt, 'phases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, concurrency in [('new_prefix', 1), ('repeated_prefix', 1), ('concurrent_repeat', 4)]:
        before = metrics(args.url)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: request(args.url, prompt), range(concurrency)))
        elapsed = time.perf_counter() - start
        # Engine metric emission can lag the final response slightly.
        time.sleep(2)
        after = metrics(args.url)
        phase = {'name': name, 'concurrency': concurrency, 'request_window_seconds': elapsed,
                 'requests': results, 'metrics_before': before, 'metrics_after': after,
                 'engine_interval': summarize(after, before, time.perf_counter() - start)}
        report['phases'].append(phase)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(name, [r['status'] for r in results], flush=True)
        if any(r['status'] != 'passed' for r in results):
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
