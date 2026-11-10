"""Kernel-level profile of decode steps through SGLang's built-in profiler. usage: prof.py <port> <tag>  env: L, CONCS, STEPS
For each concurrency: warm up, start a long greedy generation, profile STEPS decode steps, then sum GPU kernel time by name."""
import glob
import gzip
import json
import os
import sys
import threading
import time
import urllib.request
from collections import defaultdict

from transformers import AutoTokenizer

port, tag = sys.argv[1], sys.argv[2]
L = int(os.environ.get("L", "16000")); CONCS = [int(x) for x in os.environ.get("CONCS", "1,64").split(",")]
STEPS = int(os.environ.get("STEPS", "40"))
tok = AutoTokenizer.from_pretrained("/w/model")
rows = sorted((json.loads(l) for l in open("/w/aime/dense.jsonl")), key=lambda r: -r["tokens"])
pool = [tok(r["text"], add_special_tokens=False)["input_ids"][:L] for r in rows[:200]]
pool = [p for p in pool if len(p) == L]


def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def gen(prompts, n):
    ts = [threading.Thread(target=post, args=("/generate", {"input_ids": p, "sampling_params": {"temperature": 0, "max_new_tokens": n, "ignore_eos": True}})) for p in prompts]
    [t.start() for t in ts]
    return ts


for C in CONCS:
    prompts = pool[:C]
    [t.join() for t in gen(prompts, 64)]            # prefill cached, graphs warm
    out = f"/w/res/prof/{tag}-c{C}"
    os.makedirs(out, exist_ok=True)
    ts = gen(prompts, 900)
    time.sleep(1.5 if C == 1 else 4.0)              # everyone is decoding by now
    post("/start_profile", {"output_dir": out, "num_steps": STEPS, "activities": ["CPU", "GPU"]})
    [t.join() for t in ts]
    time.sleep(3)
    files = sorted(glob.glob(out + "/*.json*"), key=os.path.getmtime)
    if not files:
        print(f"{tag} C={C}: no trace written"); continue
    f = files[-1]
    ev = json.load(gzip.open(f) if f.endswith(".gz") else open(f))["traceEvents"]
    ker = defaultdict(lambda: [0.0, 0]); cpu = defaultdict(lambda: [0.0, 0]); t0, t1 = 1e30, 0
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "kernel":
            k = ker[e["name"][:70]]; k[0] += e["dur"]; k[1] += 1
            t0 = min(t0, e["ts"]); t1 = max(t1, e["ts"] + e["dur"])
        elif cat in ("gpu_memcpy", "gpu_memset"):
            k = ker[cat + " " + e["name"][:40]]; k[0] += e["dur"]; k[1] += 1
        elif cat == "cuda_runtime":
            k = cpu[e["name"][:50]]; k[0] += e["dur"]; k[1] += 1
    tot = sum(v[0] for v in ker.values()); n = sum(v[1] for v in ker.values())
    print(f"##### {tag} C={C} L={L}: {STEPS} steps asked, kernel span {(t1 - t0) / 1000:.1f} ms, GPU kernel time {tot / STEPS / 1000:.3f} ms/step, {n / STEPS:.0f} kernels/step")
    for name, (d, c) in sorted(ker.items(), key=lambda x: -x[1][0])[:16]:
        print(f"   {d / STEPS / 1000:7.3f} ms/step  {c / STEPS:6.1f}/step  {d / c:7.1f} us each  {name}")
    print("   CUDA runtime calls on CPU (inclusive, may include waiting for the GPU):")
    for name, (d, c) in sorted(cpu.items(), key=lambda x: -x[1][0])[:6]:
        print(f"   {d / STEPS / 1000:7.3f} ms/step  {c / STEPS:6.1f}/step  {name}")
    sys.stdout.flush()
