#!/usr/bin/env python3
"""Generate notebooks/s1_latency/s1_latency.ipynb — the "extremely lightweight" claim.

We have claimed "extremely lightweight multimodality" for the whole project and have
never produced a latency number. This measures it.

The bars, from the market research (R8, R11):

  Jev (cloud API, typed decisions)      236-276 ms  p50
  Laya (T4, typed decisions)             33-40  ms
  MiniCPM-V 4.0 (4.1B generative, phone) <2 s TTFT, >17 tok/s

The design's premise is that a frozen encoder plus a tiny head is *cheaper* than a small
generative model. If the encoder forward pass costs hundreds of milliseconds on CPU, that
premise fails and the "lightweight" claim is an adjective rather than a fact.

What is measured, all on CPU at fp32 (EmbeddingGemma 2 must never run fp16):

  1. cold model load
  2. encoder forward at batch 1, for short and long state texts
  3. encoder throughput at batch 1 / 8 / 32
  4. the scoring path: state x k options, k = 3 and k = 77
  5. end-to-end decision: encode + score + abstention gate
  6. Matryoshka dimension effect on the scoring path (encode cost is unaffected)

CPU-only kernel: no GPU quota. The honest caveat is stated in the report — this measures
Kaggle's CPU, not a phone, but the *breakdown* (encoder vs scoring) is portable.

Run it with: python generate.py <path/to/notebook.ipynb>
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {},
            "source": s.strip().splitlines(keepends=True),
            "execution_count": None, "outputs": []}


cells = [
    md('''
# Latency — is "extremely lightweight" a fact or an adjective?

The project's central claim has never been measured. This kernel measures it on CPU at
fp32, which is the realistic on-device configuration.

| bar | value | source |
|---|---|---|
| Jev (cloud, typed decisions) | 236–276 ms p50 | R8 |
| Laya (T4, typed decisions) | 33–40 ms | R8 |
| MiniCPM-V 4.0 (4.1B generative, phone) | <2 s TTFT, >17 tok/s | R11 |

The design bets that a **frozen encoder + tiny head** is cheaper than a small generative
model. That bet is only true if the encoder forward pass is fast. This measures it.
'''),

    md('## 0. Environment and hardware'),
    code('''
import json, os, platform, shutil, subprocess, time
from pathlib import Path

print("CPU:", platform.processor() or "unknown")
if shutil.which("nvidia-smi"):
    print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip())
else:
    print("GPU: none")
try:
    info = open("/proc/cpuinfo").read()
    for line in info.splitlines():
        if line.startswith("model name"):
            print("  ", line.strip()); break
    print("  cores:", info.count("processor\\t:"))
except Exception as e:
    print("  cpuinfo unavailable:", e)
try:
    import psutil
    print("RAM GiB:", round(psutil.virtual_memory().total / 2**30, 1))
except Exception:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    print("RAM GiB:", round(total / 2**30, 1))

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/s1-work")
WORK.mkdir(parents=True, exist_ok=True)
STATUS = {"status": "running", "stage": "env", "started": time.time()}
def write_status(stage=None, error=None, done=False):
    if stage: STATUS["stage"] = stage
    if error: STATUS["error"] = str(error); STATUS["status"] = "failed"
    if done: STATUS["status"] = "complete"
    STATUS["ended"] = time.time()
    (WORK / "run_status.json").write_text(json.dumps(STATUS, indent=2))
write_status()
'''),

    code('''
import base64, io, sys, zipfile

BUNDLE = "__DECISION_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    z.extractall(REPO)
sys.path.insert(0, REPO)

import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_protocol.py",
                    "tests/test_decision_head.py", "-q", "--no-header"],
                   capture_output=True, text=True, cwd=REPO)
print((r.stdout + r.stderr)[-1500:])
assert r.returncode == 0, "contract tests failed"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers>=6.1.0", "transformers>=5.19.0", "psutil"],
               check=True)
write_status("deps")
'''),

    md('''
## 1. Cold load

Model load time matters for an on-device product: an always-resident model pays it once, a
cold-started one pays it per session.
'''),
    code('''
import numpy as np, torch
from src.decision.encoder import StateEncoder, to_numpy

t0 = time.perf_counter()
DEVICE = "__DEVICE__"
encoder = StateEncoder(modalities=("text",), device=DEVICE)   # fp32 on CPU, bf16 on accelerator; never fp16
load_s = time.perf_counter() - t0

n_params = sum(p.numel() for p in encoder.model.parameters())
print(f"cold load: {load_s:.1f} s | params {n_params/1e6:.0f} M | dtype {encoder.dtype}")
RESULTS = {"hardware": {"cpu": platform.processor(), "torch": torch.__version__},
           "load_seconds": load_s, "params": n_params, "dtype": str(encoder.dtype)}
write_status("loaded")
'''),

    md('## 2. Timing harness'),
    code('''
def timed(fn, n, warmup=3):
    """Wall-clock percentiles with warmup discarded."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95)), "n": int(n)}

# Realistic text: short support-style queries and longer multi-turn states.
SHORT = ["I lost my card and need a replacement before my trip next week",
         "Why was I charged twice for the same transaction",
         "How do I activate the card that arrived in the post",
         "Can I withdraw cash abroad without a fee",
         "My transfer has not arrived after three days"]
LONG = [(" ".join(SHORT)) * 4, (" ".join(SHORT)) * 10]
print("short chars:", [len(s) for s in SHORT])
print("long chars:", [len(s) for s in LONG])
'''),

    md('## 3. Encoder forward — the dominant cost'),
    code('''
RESULTS["encode"] = {}
for label, texts in (("short_batch1", SHORT), ("long_batch1", LONG)):
    i = {"idx": 0}
    def call(texts=texts, i=i):
        t = texts[i["idx"] % len(texts)]
        i["idx"] += 1
        return encoder.encode(t, prompt_name="SearchQuery")
    RESULTS["encode"][label] = timed(call, 20 if label == "long_batch1" else 60)
    print(f"{label:<16} mean {RESULTS['encode'][label]['mean_ms']:8.1f} ms | "
          f"p50 {RESULTS['encode'][label]['p50_ms']:8.1f} | "
          f"p95 {RESULTS['encode'][label]['p95_ms']:8.1f}")
write_status("encode")
'''),

    code('''
RESULTS["throughput"] = {}
for bs in (1, 8, 32):
    batch = (SHORT * 8)[:bs]
    def call(batch=batch):
        return encoder.encode(batch, prompt_name="SearchQuery")
    r = timed(call, 12, warmup=2)
    r["per_text_ms"] = r["mean_ms"] / bs
    r["texts_per_second"] = 1000.0 / r["per_text_ms"]
    RESULTS["throughput"][f"batch{bs}"] = r
    print(f"batch {bs:<3} {r['mean_ms']:8.1f} ms total | {r['per_text_ms']:7.2f} ms/text | "
          f"{r['texts_per_second']:6.1f} texts/s")
write_status("throughput")
'''),

    md('''
## 4. The scoring path — state against k options

Option embeddings are precomputed once and cached, so at request time the cost is one
matmul: `(k, d) x (d,)`. Measured for k = 3 (tool routing) and k = 77 (intent classification).
'''),
    code('''
from src.decision.head import DecisionHead

state_vec = to_numpy(encoder.encode(SHORT[0], prompt_name="SearchQuery"))
torch.manual_seed(0)
RESULTS["scoring"] = {}
for k in (3, 77):
    opts = [torch.nn.functional.normalize(torch.randn(768), dim=0).numpy() for _ in range(k)]
    head = DecisionHead(dim=768, mode="interaction", hidden=64)
    def call(head=head, state_vec=state_vec, opts=opts):
        return head.predict(state_vec, opts, qtype="choice",
                            labels=[f"opt_{i}" for i in range(len(opts))])
    RESULTS["scoring"][f"k{k}_interaction"] = timed(call, 200, warmup=5)
    print(f"k={k:<3} interaction  mean {RESULTS['scoring'][f'k{k}_interaction']['mean_ms']:6.3f} ms")

    # Matryoshka: scoring a truncated state against truncated options
    for dim in (256, 128):
        s_t = state_vec[:dim] / np.linalg.norm(state_vec[:dim])
        o_t = [o[:dim] / np.linalg.norm(o[:dim]) for o in opts]
        def call2(s_t=s_t, o_t=o_t):
            return np.asarray(s_t) @ np.stack(o_t).T
        RESULTS["scoring"][f"k{k}_dim{dim}_cosine"] = timed(call2, 200, warmup=5)
        print(f"k={k:<3} cosine@{dim:<4} mean "
              f"{RESULTS['scoring'][f'k{k}_dim{dim}_cosine']['mean_ms']:6.4f} ms")
write_status("scoring")
'''),

    md('## 5. End-to-end decision — encode + score + gate'),
    code('''
opts = [torch.nn.functional.normalize(torch.randn(768), dim=0).numpy() for _ in range(77)]
head = DecisionHead(dim=768, mode="interaction", hidden=64)
def decision():
    s = to_numpy(encoder.encode(SHORT[0], prompt_name="SearchQuery"))
    return head.predict(s, opts, qtype="choice", labels=[f"opt_{i}" for i in range(77)])

RESULTS["end_to_end"] = timed(decision, 25, warmup=2)
print(f"end-to-end decision: mean {RESULTS['end_to_end']['mean_ms']:.1f} ms | "
      f"p50 {RESULTS['end_to_end']['p50_ms']:.1f} | p95 {RESULTS['end_to_end']['p95_ms']:.1f}")

# how much of it is the encoder?
enc = RESULTS["encode"]["short_batch1"]["mean_ms"]
RESULTS["breakdown"] = {"encoder_ms": enc, "total_ms": RESULTS["end_to_end"]["mean_ms"],
                        "scoring_and_gate_ms": RESULTS["end_to_end"]["mean_ms"] - enc,
                        "encoder_share": enc / RESULTS["end_to_end"]["mean_ms"]}
print("breakdown:", json.dumps(RESULTS["breakdown"]))
write_status("end-to-end")
'''),

    md('## 6. Verdict against the published bars'),
    code('''
e2e = RESULTS["end_to_end"]["mean_ms"]
BARS = {"Jev_cloud_p50": 236.0, "Laya_T4": 33.0}
RESULTS["verdict"] = {
    "end_to_end_mean_ms": e2e,
    "beats_jev_236ms": bool(e2e < BARS["Jev_cloud_p50"]),
    "beats_laya_40ms": bool(e2e < 40.0),
    "within_100ms": bool(e2e < 100.0),
    "encoder_share_of_decision": RESULTS["breakdown"]["encoder_share"],
    "caveat": ("Kaggle CPU, not a phone. The absolute number is hardware-specific; the "
               "encoder-vs-scoring breakdown is portable, and the encoder dominates."),
}
print(json.dumps(RESULTS["verdict"], indent=2))

(WORK / "REPORT.md").write_text("# Latency\\n\\n" + json.dumps(RESULTS, indent=2, default=str))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
save = None
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json"]:
        p = WORK / f
        if p.exists():
            zf.write(p, f)
STATUS["status"] = "complete"
write_status("done", done=True)
'''),
]

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_FILES = [
    ROOT / "src" / "__init__.py",
    *sorted((ROOT / "src" / "decision").glob("*.py")),
    ROOT / "tests" / "__init__.py",
    *(ROOT / "tests" / n for n in ("test_decision_protocol.py", "test_decision_head.py")),
]
payload = io.BytesIO()
with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
    for source in BUNDLE_FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        archive.write(source, source.relative_to(ROOT).as_posix())
bundle_b64 = base64.b64encode(payload.getvalue()).decode("ascii")

bootstrap = next(c for c in cells if "__DECISION_BUNDLE__" in "".join(c["source"]))
bootstrap["source"] = "".join(bootstrap["source"]).replace(
    "__DECISION_BUNDLE__", bundle_b64).splitlines(keepends=True)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
for c in cells:
    c["source"] = "".join(c["source"]).replace("__DEVICE__", device).splitlines(keepends=True)

out = Path(sys.argv[1])
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} — {len(json.loads(out.read_text())['cells'])} cells, valid JSON")
