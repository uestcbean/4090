"""Local observability UI: scrape vLLM, never invent missing metrics."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

ROOT = Path(__file__).parent
STATE = {"ready": False, "error": "Waiting for first scrape", "history": []}
LOCK = threading.Lock()


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([^\s{]+)(?:\{.*\})?\s+([^\s]+)', line)
        if match:
            try:
                val = float(match[2])
                if math.isfinite(val):
                    values[match[1]] = values.get(match[1], 0) + val
            except ValueError:
                pass
    return values


def summarize(current, previous, seconds):
    def get(*names):
        return next((current[n] for n in names if n in current), None)
    def delta(name):
        if name not in current or name not in previous or current[name] < previous[name]:
            return None
        return current[name] - previous[name]
    def rate(name):
        d = delta(name)
        return d / seconds if d is not None and seconds > 0 else None
    def mean(base):
        count, total = delta(base + "_count"), delta(base + "_sum")
        return total / count if count and total is not None else None
    hits, queries = delta("vllm:prefix_cache_hits_total"), delta("vllm:prefix_cache_queries_total")
    return {"running": get("vllm:num_requests_running"),
            "waiting": get("vllm:num_requests_waiting"),
            "completed": get("vllm:request_success_total"),
            "requests_per_second": rate("vllm:request_success_total"),
            "output_tokens_per_second": rate("vllm:generation_tokens_total"),
            "ttft_seconds": mean("vllm:time_to_first_token_seconds"),
            "tpot_seconds": mean("vllm:request_time_per_output_token_seconds"),
            "itl_seconds": mean("vllm:inter_token_latency_seconds"),
            "kv_usage": get("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
            "prefix_hit_rate": hits / queries if hits is not None and queries else None,
            "prefix_queries_delta": queries}


def poll(engine):
    previous, previous_time = {}, None
    history = deque(maxlen=120)
    while True:
        now = time.time()
        snapshot = {"time": now, "ready": False, "engine": engine}
        try:
            with urllib.request.urlopen(engine + "/metrics", timeout=3) as response:
                raw = response.read().decode()
            current = parse_metrics(raw)
            if not any(n.startswith("vllm:") for n in current):
                raise ValueError("Endpoint returned no vLLM metrics")
            snapshot.update(summarize(current, previous, now - previous_time if previous_time else 0))
            snapshot["ready"] = True
            snapshot["metric_names"] = sorted(current)
            previous, previous_time = current, now
        except Exception as exc:
            snapshot["error"] = str(exc)
            previous, previous_time = {}, None
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            if p.returncode == 0:
                fields = p.stdout.strip().splitlines()[0].split(",")
                snapshot["gpu"] = dict(zip(["busy", "used_mib", "total_mib", "temperature", "watts"],
                    [float(x) if re.match(r'^\s*[\d.]+\s*$', x) else None for x in fields]))
        except Exception:
            pass
        history.append({k: snapshot.get(k) for k in ["time", "output_tokens_per_second", "ttft_seconds"]})
        snapshot["history"] = list(history)
        with LOCK:
            STATE.clear()
            STATE.update(snapshot)
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            with LOCK:
                body = json.dumps(STATE, allow_nan=False).encode()
            mime = "application/json"
        elif self.path in ("/", "/index.html"):
            body, mime = (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    threading.Thread(target=poll, args=(args.engine,), daemon=True).start()
    print(f"Observability: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
