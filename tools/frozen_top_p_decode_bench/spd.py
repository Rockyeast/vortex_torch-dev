"""Decode speed at a fixed context length: wall(N_LONG new tokens) - wall(N_SHORT new tokens), same prompts, prompts cached.

usage: spd.py <port> <tag>   env: LENS=4096,8192,16384 CONCS=1,16,64 REPEATS=2
Prompts are the first L tokens of the student's own dense AIME responses (aime/dense.jsonl), greedy, ignore_eos.
"""
import json
import os
import sys
import threading
import time
import urllib.request

from transformers import AutoTokenizer

port, tag = sys.argv[1], sys.argv[2]
LENS = [int(x) for x in os.environ.get("LENS", "4096,8192,16384").split(",")]
CONCS = [int(x) for x in os.environ.get("CONCS", "1,16,64").split(",")]
REPEATS = int(os.environ.get("REPEATS", "2"))
N_SHORT, N_LONG = 64, 576
tok = AutoTokenizer.from_pretrained("/w/model")
rows = [json.loads(l) for l in open("/w/aime/dense.jsonl")]
rows = [r for r in rows if r["tokens"] >= 4000]
ids_all = []
seen = set()
for r in sorted(rows, key=lambda r: -r["tokens"]):
    if r["id"] in seen:  # one response per problem first, so a batch holds different texts
        continue
    seen.add(r["id"]); ids_all.append(tok(r["text"], add_special_tokens=False)["input_ids"])
for r in sorted(rows, key=lambda r: -r["tokens"]):
    ids_all.append(tok(r["text"], add_special_tokens=False)["input_ids"])


def batch_wall(prompts, n_new):
    done = [None] * len(prompts)

    def go(i):
        body = {"input_ids": prompts[i], "sampling_params": {"temperature": 0, "max_new_tokens": n_new, "ignore_eos": True}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        done[i] = d["meta_info"]["completion_tokens"]

    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    assert all(x == n_new for x in done), done
    return wall


out = []
for L in LENS:
    pool = [x[:L] for x in ids_all if len(x) >= L]
    for C in CONCS:
        if len(pool) < C:
            print(f"skip L={L} C={C}: only {len(pool)} texts", flush=True); continue
        prompts = pool[:C]
        batch_wall(prompts, N_SHORT)  # cold: prefill + warm-up, discarded
        for rep in range(REPEATS):
            ws = batch_wall(prompts, N_SHORT)
            wl = batch_wall(prompts, N_LONG)
            steps = N_LONG - N_SHORT
            ms = (wl - ws) / steps * 1000
            r = {"tag": tag, "L": L, "C": C, "rep": rep, "wall_short": round(ws, 3), "wall_long": round(wl, 3),
                 "ms_per_step": round(ms, 3), "tok_s": round(C * 1000 / ms, 1)}
            out.append(r); print(json.dumps(r), flush=True)
os.makedirs("/w/res/spd", exist_ok=True)
json.dump(out, open(f"/w/res/spd/{tag}.json", "w"), indent=1)
