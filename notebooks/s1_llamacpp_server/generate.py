#!/usr/bin/env python3
"""Generate notebooks/s1_llamacpp_server/s1_llamacpp_server.ipynb — llama.cpp, take two.

Two earlier attempts failed for an informative reason: `llama-cpp-python` (PyPI 0.3.36, and the
git master, which still reports 0.3.36) vendors a llama.cpp revision that predates
`embeddinggemma-2`. The GGUF was published the same day as the model, so the bindings cannot load
it — `Failed to load model from file`.

This version builds **llama.cpp itself** from `ggml-org/llama.cpp` master — the same org that
published the GGUF — and drives it through `llama-server`, which is also the actual production
pattern (one resident process, requests over HTTP) rather than a Python binding.

Measures, all CPU:
  1. build time
  2. does the GGUF load at all (with a clear diagnostic if the architecture is unsupported)
  3. thread scaling via `-t 1..N`
  4. batch-1 latency, short and long states, over HTTP
  5. throughput + QUALITY: Banking77 zero-shot accuracy under Q8_0 vs the PyTorch 92.92%
  6. drift: Q8_0 vs BF16

CPU-only: no GPU quota.

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
# llama.cpp, take two — build the engine, not the bindings

Two attempts established something worth knowing: **`llama-cpp-python` cannot load
EmbeddingGemma 2 GGUFs.** PyPI's 0.3.36 and even the git master (which still reports 0.3.36)
vendor a llama.cpp revision that predates the architecture, so the official GGUF fails with
`Failed to load model from file`.

This builds **`ggml-org/llama.cpp` master** — the same org that published the GGUF — and drives it
via `llama-server`, which is the real production pattern: one resident process, HTTP requests,
no Python binding in the hot path.

| | our PyTorch fp32 | GGUF Q8_0 |
|---|---:|---:|
| weights | 1.08 GB | **0.31 GB** |
| CPU bandwidth floor | 54.2 ms | **15.5 ms** |

Measures: build time, whether the GGUF loads at all, thread scaling, batch-1 latency (short and
long), throughput, **zero-shot accuracy under Q8_0** against the known 92.92%, and Q8-vs-BF16
drift. Task prefixes are applied manually — sentence-transformers adds them, llama.cpp does not.
'''),

    md('## 0. Environment'),
    code('''
import json, os, platform, shutil, subprocess, time
from pathlib import Path

print("CPU:", platform.processor() or "unknown")
try:
    info = open("/proc/cpuinfo").read()
    for line in info.splitlines():
        if line.startswith("model name"):
            print("  ", line.strip()); break
    print("   cores:", info.count("processor\\t:"))
except Exception:
    pass

import numpy as np
print("numpy", np.__version__)

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

BUNDLE = "__SOURCE_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    z.extractall(REPO)
sys.path.insert(0, REPO)

r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_protocol.py",
                    "-q", "--no-header"], capture_output=True, text=True, cwd=REPO)
print((r.stdout + r.stderr)[-1000:])
assert r.returncode == 0, "protocol tests failed"
write_status("bundle")
'''),

    md('## 1. Build llama.cpp from master'),
    code('''
t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja",
                "huggingface_hub"], check=True)

# NOT under /kaggle/working: the source+build tree would become a kernel artifact
# and make `kaggle kernels output` download gigabytes.
SRC = "/tmp/llama.cpp"
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/ggml-org/llama.cpp.git", SRC],
               check=True, capture_output=True, text=True)
head = subprocess.run(["git", "-C", SRC, "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("llama.cpp commit:", head)

cfg = subprocess.run(["cmake", "-B", "build", "-DGGML_NATIVE=ON", "-DGGML_OPENMP=ON",
                      "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"],
                     cwd=SRC, capture_output=True, text=True)
print("cmake configure rc:", cfg.returncode)
if cfg.returncode != 0:
    print((cfg.stdout + cfg.stderr)[-2500:])

bld = subprocess.run(["cmake", "--build", "build", "--config", "Release",
                      "--target", "llama-server", "llama-embedding", "-j", "4"],
                     cwd=SRC, capture_output=True, text=True)
build_s = time.perf_counter() - t0
print(f"build: {build_s:.0f} s | rc {bld.returncode}")
if bld.returncode != 0:
    print((bld.stdout + bld.stderr)[-3000:])

bins = sorted(str(p) for p in Path(SRC, "build", "bin").glob("*") if p.is_file())
print("binaries:", [Path(b).name for b in bins][:12])
RESULTS = {"llama_cpp_commit": head, "build_seconds": build_s, "build_rc": bld.returncode}
write_status("built")
'''),

    md('## 2. Fetch the GGUFs'),
    code('''
from huggingface_hub import hf_hub_download

REPO_GGUF = "ggml-org/embeddinggemma-2-GGUF"
paths = {}
for fn in ("embeddinggemma-2-Q8_0.gguf", "embeddinggemma-2-BF16.gguf"):
    paths[fn] = hf_hub_download(REPO_GGUF, fn)
    print(f"{fn:<34} {os.path.getsize(paths[fn])/1e6:8.1f} MB")
RESULTS["gguf_sizes_mb"] = {k: os.path.getsize(v) / 1e6 for k, v in paths.items()}
write_status("downloaded")
'''),

    md('''
## 3. Does it load? (the question the bindings could not answer)

Run the CLI once and show llama.cpp's own diagnostics. If the architecture is unsupported this
prints the reason instead of a bare `ValueError`.
'''),
    code('''
EMBED_BIN = str(Path(SRC, "build", "bin", "llama-embedding"))
print("using:", EMBED_BIN, "| exists:", os.path.exists(EMBED_BIN))

probe = subprocess.run([EMBED_BIN, "-m", paths["embeddinggemma-2-Q8_0.gguf"],
                        "-p", "task: search result | query: hello world",
                        "-t", "4", "--embd-normalize", "2"],
                       capture_output=True, text=True, timeout=600)
print("rc:", probe.returncode)
out = (probe.stdout or "") + (probe.stderr or "")
print(out[-2500:])
RESULTS["cli_probe_rc"] = probe.returncode
assert probe.returncode == 0, "llama.cpp master cannot load this GGUF either — see diagnostics above"
write_status("probe-ok")
'''),

    md('## 4. Serve it and measure'),
    code('''
import socket, urllib.request

SERVER_BIN = str(Path(SRC, "build", "bin", "llama-server"))
PREFIX_QUERY = "task: search result | query: "   # from EG2's config_sentence_transformers.json
PREFIX_DOC = "title: none | text: "

def wait_ready(port, timeout=600):
    """Poll /health, not the port.

    A previous attempt used a socket connect, which succeeds as soon as the listener is
    up — *before* the model finishes loading — and the first request then failed with
    HTTP 503 Service Unavailable. llama-server reports readiness on /health.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def start_server(gguf, threads, port):
    p = subprocess.Popen([SERVER_BIN, "-m", gguf, "--embeddings", "--port", str(port),
                          "-t", str(threads), "-c", "2048", "--no-webui"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_ready(port), f"server on {port} never became ready"
    return p

def embed(text, port, retries=30):
    body = json.dumps({"input": text, "model": "eg2"}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
                                     data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                vec = json.loads(resp.read())["data"][0]["embedding"]
            break
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries - 1:   # still loading; back off
                time.sleep(1)
                continue
            raise
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def timed(fn, n, warmup=3):
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

SHORT = ["I lost my card and need a replacement before my trip next week",
         "Why was I charged twice for the same transaction",
         "How do I activate the card that arrived in the post"]
LONG = [(" ".join(SHORT)) * 10]
print("short chars:", [len(s) for s in SHORT], "| long chars:", [len(s) for s in LONG])
write_status("server-ready")
'''),

    md('## 5. Thread scaling'),
    code('''
n_cpu = os.cpu_count() or 4
RESULTS["threads"] = {}
procs = {}
for nt in sorted({1, 2, 4, n_cpu}):
    port = 8100 + nt
    procs[nt] = start_server(paths["embeddinggemma-2-Q8_0.gguf"], nt, port)
    i = {"n": 0}
    def call(port=port, i=i):
        t = SHORT[i["n"] % len(SHORT)]; i["n"] += 1
        return embed(PREFIX_QUERY + t, port)
    r = timed(call, 12, warmup=2)
    RESULTS["threads"][f"n_threads={nt}"] = r
    print(f"n_threads={nt:<3} mean {r['mean_ms']:8.1f} ms | p50 {r['p50_ms']:8.1f} | p95 {r['p95_ms']:8.1f}",
          flush=True)
for nt, p in procs.items():
    if nt != n_cpu:
        p.terminate()
write_status("threads")
'''),

    md('## 6. Batch-1 latency: Q8_0 vs BF16'),
    code('''
PORT = 8100 + n_cpu
RESULTS["encode"] = {}
RESULTS["load_seconds"] = {}

def measure(tag, gguf):
    t0 = time.perf_counter()
    proc = start_server(gguf, n_cpu, PORT)
    RESULTS["load_seconds"][tag] = time.perf_counter() - t0
    i = {"n": 0}
    def short_call(i=i):
        t = SHORT[i["n"] % len(SHORT)]; i["n"] += 1
        return embed(PREFIX_QUERY + t, PORT)
    RESULTS["encode"][f"{tag}_short"] = timed(short_call, 15, warmup=3)
    j = {"n": 0}
    def long_call(j=j):
        t = LONG[j["n"] % len(LONG)]; j["n"] += 1
        return embed(PREFIX_QUERY + t, PORT)
    RESULTS["encode"][f"{tag}_long"] = timed(long_call, 6, warmup=2)
    for k in (f"{tag}_short", f"{tag}_long"):
        v = RESULTS["encode"][k]
        print(f"{k:<12} mean {v['mean_ms']:8.1f} ms | p50 {v['p50_ms']:8.1f} | p95 {v['p95_ms']:8.1f}")
    print(f"{tag} server startup: {RESULTS['load_seconds'][tag]:.2f} s")
    return proc

proc_q8 = measure("q8_0", paths["embeddinggemma-2-Q8_0.gguf"])
proc_q8.terminate(); time.sleep(3)
proc_bf16 = measure("bf16", paths["embeddinggemma-2-BF16.gguf"])
write_status("encode")
'''),

    md('''
## 7. Quality under Q8_0 — is it still the same model?

Banking77 zero-shot intent routing, same task and same prefixes as the PyTorch run that gave
**92.92%**.
'''),
    code('''
import ast, requests, pandas as pd

RAW = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
script = requests.get("https://huggingface.co/datasets/PolyAI/banking77/resolve/main/banking77.py").text
i, j = script.index("names=["), script.index("]", script.index("names=["))
label_names = ast.literal_eval(script[i + 6:j + 1])
test_df = pd.read_csv(RAW + "test.csv")
option_texts = [n.replace("_", " ") for n in label_names]
text_col, label_col = test_df.columns[0], test_df.columns[1]
test_texts = test_df[text_col].tolist()
test_y = np.array([label_names.index(v) if isinstance(v, str) else int(v)
                   for v in test_df[label_col]])
print(f"test {len(test_texts)} | classes {len(option_texts)}")

t0 = time.perf_counter()
O_q8 = np.stack([embed(PREFIX_DOC + o, PORT) for o in option_texts])
S_q8 = np.stack([embed(PREFIX_QUERY + t, PORT) for t in test_texts])
enc_s = time.perf_counter() - t0
n_texts = len(S_q8) + len(O_q8)
RESULTS["banking77_throughput"] = {"texts": n_texts, "seconds": enc_s,
                                   "texts_per_second": n_texts / enc_s}

from src.decision import protocol
scores = S_q8 @ O_q8.T
K = len(option_texts)
targets = np.zeros((len(test_y), K)); targets[np.arange(len(test_y)), test_y] = 1.0
tau = protocol.fit_temperature(scores, targets)
block = protocol.metric_block(protocol.softmax(scores, tau), targets)
RESULTS["banking77_quality_q8"] = dict(block, temperature=tau,
                                       pytorch_reference_accuracy=0.9292,
                                       accuracy_delta=block["accuracy"] - 0.9292)
print(f"throughput: {RESULTS['banking77_throughput']['texts_per_second']:.1f} texts/s")
print(f"zero-shot accuracy under Q8_0: {block['accuracy']:.4f} "
      f"(PyTorch reference 0.9292, delta {block['accuracy'] - 0.9292:+.4f})")
print(f"log {block['log_score']:.3f} | brier {block['brier']:.3f} | ece {block['ece_report_only']:.3f}")
write_status("quality")
'''),

    md('## 8. Drift: Q8_0 vs BF16, and verdict'),
    code('''
sample = np.arange(0, len(test_texts), max(1, len(test_texts) // 150))[:150]
S_bf16 = np.stack([embed(PREFIX_QUERY + test_texts[i], PORT) for i in sample])
cos = (S_bf16 * S_q8[sample]).sum(axis=1)
RESULTS["drift"] = {"n": int(len(sample)), "mean_cosine": float(cos.mean()),
                    "min_cosine": float(cos.min()), "p05_cosine": float(np.percentile(cos, 5))}
print(json.dumps(RESULTS["drift"], indent=2))
proc_bf16.terminate()

q8_short = RESULTS["encode"]["q8_0_short"]["mean_ms"]
q8_long = RESULTS["encode"]["q8_0_long"]["mean_ms"]
PYT_CPU_SHORT, PYT_CPU_LONG, PYT_CPU_THROUGHPUT = 363.5, 6595.0, 1000.0 / 416.1
thr = RESULTS["banking77_throughput"]["texts_per_second"]
RESULTS["verdict"] = {
    "q8_short_ms": q8_short, "pytorch_cpu_short_ms": PYT_CPU_SHORT,
    "short_speedup": PYT_CPU_SHORT / q8_short,
    "q8_long_ms": q8_long, "pytorch_cpu_long_ms": PYT_CPU_LONG,
    "long_speedup": PYT_CPU_LONG / q8_long,
    "q8_throughput_texts_per_s": thr, "pytorch_cpu_texts_per_s": PYT_CPU_THROUGHPUT,
    "throughput_speedup": thr / PYT_CPU_THROUGHPUT,
    "quality_delta_vs_pytorch": RESULTS["banking77_quality_q8"]["accuracy_delta"],
    "quality_preserved": bool(RESULTS["banking77_quality_q8"]["accuracy_delta"] > -0.01),
    "thread_scaling_1_to_max": (RESULTS["threads"]["n_threads=1"]["mean_ms"] /
                                RESULTS["threads"][f"n_threads={n_cpu}"]["mean_ms"]
                                if f"n_threads={n_cpu}" in RESULTS["threads"] else None),
}
print(json.dumps(RESULTS["verdict"], indent=2))

(WORK / "REPORT.md").write_text("# llama.cpp server path\\n\\n" + json.dumps(RESULTS, indent=2, default=str))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
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
    ROOT / "tests" / "test_decision_protocol.py",
]
payload = io.BytesIO()
with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
    for source in BUNDLE_FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        archive.write(source, source.relative_to(ROOT).as_posix())
bundle_b64 = base64.b64encode(payload.getvalue()).decode("ascii")

bootstrap = next(c for c in cells if "__SOURCE_BUNDLE__" in "".join(c["source"]))
bootstrap["source"] = "".join(bootstrap["source"]).replace(
    "__SOURCE_BUNDLE__", bundle_b64).splitlines(keepends=True)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(sys.argv[1])
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} — {len(json.loads(out.read_text())['cells'])} cells, valid JSON")
