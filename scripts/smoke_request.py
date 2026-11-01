"""One localhost streaming request. Content-chunk timing is not per-token timing."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = {"model": args.model, "messages": [{"role": "user", "content":
            "请用一句话解释什么是 KV Cache。"}], "max_tokens": 128,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    result = {"request": body, "status": "failed", "usage": None}
    start = time.perf_counter()
    chunks = []
    first = None
    done = False
    try:
        req = urllib.request.Request(args.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        first = first if first is not None else time.perf_counter() - start
                        chunks.append(content)
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
        if not done or not chunks or not result.get("finish_reason"):
            raise RuntimeError("Incomplete or empty stream")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update(content="".join(chunks), first_content_chunk_seconds=first,
                  elapsed_seconds=time.perf_counter() - start,
                  note="Single smoke request, not a performance benchmark; no TPOT inferred from chunks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
