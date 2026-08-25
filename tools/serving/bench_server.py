"""Server-level benchmark: per-slot prefill and decode across depth and sessions.

Every prompt's token count is read from the API's own `usage`, never assumed.
An earlier version of this harness derived tokens from a words-per-token
estimate and reported rates that were wrong by up to 3x; worse, the salt used
to defeat the prefix cache was embedded per word, so changing it changed the
workload. The salt is now a single leading token and counts are measured.
"""
import itertools, json, statistics as st, sys, threading, time, urllib.request

BASE = sys.argv[1]
OUT = sys.argv[2]
SPEC = json.loads(sys.argv[3])       # {"depths":[words...], "sessions":[n...], "trials":k}
MAX_TOKENS = SPEC.get("max_tokens", 32)
_salts = itertools.count()


def fresh_salt():
    """A short, unique leading token: unique so nothing is served from the
    prefix cache, short so it does not change the prompt's token count."""
    return f"z{next(_salts)}"

def prompt_for(salt, k, words):
    return f"{salt}{k} " + " ".join(f"q{(i * 7919) % 50021}" for i in range(words))

def post(payload, timeout=3600):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def measure_tokens(words, salt):
    with post({"model": "qwen3.6-35b-a3b",
               "messages": [{"role": "user", "content": prompt_for(salt, 0, words)}],
               "max_tokens": 1, "stream": False, "temperature": 1.0}) as r:
        return json.load(r)["usage"]["prompt_tokens"]

def stream_one(salt, k, words, results, lock, t0_all):
    payload = {"model": "qwen3.6-35b-a3b",
               "messages": [{"role": "user", "content": prompt_for(salt, k, words)}],
               "max_tokens": MAX_TOKENS, "stream": True, "temperature": 1.0}
    t0 = time.perf_counter()
    ttft, last, gaps, n = None, None, [], 0
    try:
        with post(payload) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    if line[6:9] == "[DO":
                        break
                    continue
                d = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        gaps.append(now - last)
                    last, n = now, n + 1
    except Exception as exc:                                     # noqa: BLE001
        with lock:
            results[k] = {"error": repr(exc)}
        return
    with lock:
        results[k] = {"ttft_s": ttft, "out_tokens": n,
                      "median_itl_s": st.median(gaps) if gaps else None,
                      "first_token_at_s": (t0 + ttft - t0_all) if ttft else None,
                      "done_at_s": time.perf_counter() - t0_all}

rows = []
for words in SPEC["depths"]:
    ptok = measure_tokens(words, fresh_salt())
    for sessions in SPEC["sessions"]:
        for trial in range(SPEC.get("trials", 2)):
            salt = fresh_salt()
            results, lock = {}, threading.Lock()
            t0 = time.perf_counter()
            ths = [threading.Thread(target=stream_one,
                                    args=(salt, k, words, results, lock, t0))
                   for k in range(sessions)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            errs = [v["error"] for v in results.values() if "error" in v]
            ok = [v for v in results.values() if "error" not in v and v["ttft_s"]]
            if not ok:
                rows.append({"prompt_tokens": ptok, "sessions": sessions, "trial": trial,
                             "errors": errs})
                continue
            last_ttft = max(v["first_token_at_s"] for v in ok)
            itls = [v["median_itl_s"] for v in ok if v["median_itl_s"]]
            out_total = sum(v["out_tokens"] for v in ok)
            decode_window = max(v["done_at_s"] for v in ok) - last_ttft
            row = {
                "prompt_tokens": ptok, "sessions": sessions, "trial": trial,
                "wall_s": wall,
                "prefill_agg_tok_s": ptok * len(ok) / last_ttft,
                "prefill_per_slot_tok_s": ptok * len(ok) / last_ttft / len(ok),
                "decode_agg_tok_s": (out_total - len(ok)) / decode_window if decode_window > 0 else None,
                "decode_per_slot_tok_s": (1 / st.median(itls)) if itls else None,
                "ttfts": sorted(round(v["ttft_s"], 2) for v in ok),
                "errors": errs,
            }
            rows.append(row)
            print(f"  {ptok:>7} tok x {sessions} sessions  t{trial}: "
                  f"prefill {row['prefill_agg_tok_s']:6.0f} agg / "
                  f"{row['prefill_per_slot_tok_s']:6.0f} per-slot   "
                  f"decode {row['decode_agg_tok_s'] or 0:5.1f} agg / "
                  f"{row['decode_per_slot_tok_s'] or 0:4.1f} per-slot   "
                  f"ttfts {row['ttfts']}", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
