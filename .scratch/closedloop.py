"""Closed-loop client reporting req/s, TTFT, TPOT and E2E, streaming."""
import argparse, json, statistics, threading, time, urllib.request, uuid


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    i = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prompt-tokens", type=int, default=8128)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    words = ["alpha", "bravo", "delta", "echo", "gamma", "hotel", "india", "kilo"]
    lock = threading.Lock()
    ttfts, tpots, e2es = [], [], []
    errors = [0]
    counter = [0]

    def one(i):
        nonce = uuid.uuid4().hex
        # one short word is about 1.25 tokens for this tokenizer, so scale down
        # to land near the requested token count rather than overshoot the
        # model's 8192-token context.
        n_words = max(1, int(a.prompt_tokens / 1.25) - 16)
        filler = " ".join(words[(i * 7 + k) % len(words)] for k in range(n_words))
        body = json.dumps({
            "model": "qwen3-omni", "stream": True,
            "messages": [{"role": "user", "content": f"{nonce} {filler}"}],
            "max_tokens": a.max_tokens,
        }).encode()
        t0 = time.perf_counter()
        first = None
        n_tok = 0
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(url, data=body,
                                       headers={"Content-Type": "application/json"}),
                timeout=900)
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                ch = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if ch:
                    if first is None:
                        first = time.perf_counter() - t0
                    n_tok += 1
            e2e = time.perf_counter() - t0
            with lock:
                if first is None:
                    errors[0] += 1
                else:
                    ttfts.append(first)
                    e2es.append(e2e)
                    if n_tok > 1:
                        tpots.append((e2e - first) / (n_tok - 1))
        except Exception:
            with lock:
                errors[0] += 1

    def worker():
        while True:
            with lock:
                if counter[0] >= a.n:
                    return
                i = counter[0]
                counter[0] += 1
            one(i)

    start = time.perf_counter()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=1800)
    wall = time.perf_counter() - start

    print(json.dumps({
        "concurrency": a.concurrency, "n": a.n, "ok": len(ttfts), "errors": errors[0],
        "wall_s": round(wall, 2),
        "req_per_s": round(len(ttfts) / wall, 4) if wall else None,
        "ttft_p50": round(pct(ttfts, 50) or 0, 3), "ttft_p95": round(pct(ttfts, 95) or 0, 3),
        "tpot_p50": round(pct(tpots, 50) or 0, 4),
        "e2e_p50": round(pct(e2es, 50) or 0, 3), "e2e_p95": round(pct(e2es, 95) or 0, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
