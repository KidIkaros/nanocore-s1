#!/usr/bin/env python3
"""Generate notebooks/s1_llamacpp/s1_llamacpp.ipynb — the llama.cpp path.

Answers the question the latency run raised: our PyTorch path measured 416 ms per decision on
CPU, 7.7x above its own memory-bandwidth floor, at ~27-30 GFLOPS at every batch size (no
parallelism benefit). llama.cpp attacks both causes — 3.5x less weight traffic with Q8_0, and a
SIMD-optimized, explicitly threaded C++ runtime.

Official GGUFs of EmbeddingGemma 2 exist in the llama.cpp project's own org
(ggml-org/embeddinggemma-2-GGUF): BF16 558 MB, Q8_0 310 MB.

Measures, all CPU (where the gap is):
  1. build + load time
  2. thread scaling, n_threads = 1..N  (is the gap thread starvation?)
  3. batch-1 latency, short and long states
  4. throughput over the real Banking77 test set
  5. QUALITY: zero-shot accuracy under Q8_0, against the known PyTorch 92.92%
  6. drift: Q8_0 vs BF16 embeddings on a fixed sample

Prompt handling matters for a fair comparison: sentence-transformers applies task prefixes
that llama.cpp does not, so the same prefixes are applied manually here.

Precision: the model card forbids fp16 because EG2's activations exceed fp16's dynamic range,
so the F16 GGUF is avoided; BF16 (fp32 exponent range) and Q8_0 (integer weights, fp32
activations) are the safe builds.

CPU-only kernel: no GPU quota.

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
# The llama.cpp path — does the 7.7× CPU gap close?

Our PyTorch path measured **416 ms per decision on CPU**, **7.7× above its own memory-bandwidth
floor**, at **~27–30 GFLOPS at every batch size** — no parallelism benefit at all.

llama.cpp attacks both causes at once:

| | our PyTorch fp32 | GGUF Q8_0 | GGUF BF16 |
|---|---:|---:|---:|
| weights | 1.08 GB | **0.31 GB** | 0.56 GB |
| CPU bandwidth floor | 54.2 ms | **15.5 ms** | 27.9 ms |

Official GGUFs exist in the llama.cpp project's own org: `ggml-org/embeddinggemma-2-GGUF`.

What is measured: build/load time, **thread scaling**, batch-1 latency (short and long states),
throughput on the real Banking77 test set, **zero-shot accuracy under Q8_0** against the known
PyTorch 92.92%, and **embedding drift** Q8_0 vs BF16.

Task prefixes are applied manually — sentence-transformers adds them and llama.cpp does not, so
omitting them would produce a quality difference that is not about quantization.
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
except Exception as e:
    print("  cpuinfo unavailable:", e)
print("GPU:", (subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
               if shutil.which("nvidia-smi") else "none"))

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

    md('## 1. Source bundle'),
    code('''
# The protocol module measures quality here, so it must be present and verified.
import base64, io, sys, zipfile

BUNDLE = "__SOURCE_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    z.extractall(REPO)
sys.path.insert(0, REPO)

import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_protocol.py",
                    "-q", "--no-header"], capture_output=True, text=True, cwd=REPO)
print((r.stdout + r.stderr)[-1200:])
assert r.returncode == 0, "protocol tests failed"
write_status("bundle")
'''),

    md('''
## 2. Build llama.cpp bindings

**From git master, not PyPI.** The first attempt installed `llama-cpp-python` 0.3.36 from PyPI and
the GGUF failed to load with `Failed to load model from file` — the release bundles a llama.cpp
revision that predates `embeddinggemma-2` (the GGUF was published the same day as the model).
Master tracks the ggml-org repo that produced the GGUF.

Built from source for CPU: this is the path where our 7.7× gap lives, and a CUDA build would cost
much longer to compile for a case already measured on GPU.
'''),
    code('''
import subprocess, sys, time

t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja",
                "huggingface_hub"], check=True)
env = dict(os.environ)
env["CMAKE_ARGS"] = "-DGGML_NATIVE=ON -DGGML_OPENMP=ON"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "llama-cpp-python @ git+https://github.com/abetlen/llama-cpp-python.git"],
                   env=env, capture_output=True, text=True)
build_s = time.perf_counter() - t0
print(f"build+install: {build_s:.0f} s | returncode {r.returncode}")
if r.returncode != 0:
    print((r.stdout + r.stderr)[-3000:])

import llama_cpp
print("llama_cpp", getattr(llama_cpp, "__version__", "unknown"))
try:
    print(llama_cpp.llama_print_system_info().decode()[:400])
except Exception as e:
    print("system info unavailable:", e)
RESULTS = {"build_seconds": build_s, "install_source": "git master",
           "llama_cpp_version": str(getattr(llama_cpp, "__version__", "unknown"))}
write_status("built")
'''),

    md('## 3. Fetch the GGUFs'),
    code('''
from huggingface_hub import hf_hub_download

REPO = "ggml-org/embeddinggemma-2-GGUF"
paths = {}
for fn in ("embeddinggemma-2-Q8_0.gguf", "embeddinggemma-2-BF16.gguf"):
    p = hf_hub_download(REPO, fn)
    paths[fn] = p
    print(f"{fn:<34} {os.path.getsize(p)/1e6:8.1f} MB")
RESULTS["gguf_sizes_mb"] = {k: os.path.getsize(v) / 1e6 for k, v in paths.items()}
write_status("downloaded")
'''),

    md('''
## 4. Helpers

`SearchQuery` and `Document` prefixes are the exact strings from the model's
`config_sentence_transformers.json`, applied manually so llama.cpp sees the same input
sentence-transformers would.
'''),
    code('''
from llama_cpp import Llama

# exact prefixes from google/embeddinggemma-2 config_sentence_transformers.json
PREFIX_QUERY = "task: search result | query: "
PREFIX_DOC = "title: none | text: "

def load(path, n_threads):
    return Llama(model_path=path, embedding=True, n_ctx=2048, n_threads=n_threads,
                 pooling_type=1,          # LLAMA_POOLING_TYPE_MEAN — EG2 uses mean pooling
                 verbose=False, logits_all=False)

def embed_one(llm, text):
    """One normalized embedding. Handles both llama-cpp-python embedding APIs."""
    try:
        out = llm.create_embedding(text)
        vec = out["data"][0]["embedding"]
    except Exception:
        vec = llm.embed(text)
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

# Fail loudly and specifically if this llama.cpp build does not know the architecture.
try:
    _probe = load(paths["embeddinggemma-2-Q8_0.gguf"], 2)
    print("GGUF loads OK with this llama.cpp build")
    del _probe
except Exception as e:
    print("GGUF FAILED TO LOAD:", type(e).__name__, str(e)[:300])
    try:
        print("llama.cpp build:", llama_cpp.llama_print_system_info().decode()[:300])
    except Exception:
        pass
    raise
'''),

    md('## 5. Thread scaling — is the gap thread starvation?'),
    code('''
n_cpu = os.cpu_count() or 4
print("os.cpu_count():", n_cpu)
RESULTS["threads"] = {}
for nt in sorted({1, 2, 4, n_cpu}):
    llm = load(paths["embeddinggemma-2-Q8_0.gguf"], nt)
    i = {"n": 0}
    def call(llm=llm, i=i):
        t = SHORT[i["n"] % len(SHORT)]; i["n"] += 1
        return embed_one(llm, PREFIX_QUERY + t)
    r = timed(call, 15, warmup=2)
    RESULTS["threads"][f"n_threads={nt}"] = r
    print(f"n_threads={nt:<3} mean {r['mean_ms']:8.1f} ms | p50 {r['p50_ms']:8.1f} | p95 {r['p95_ms']:8.1f}")
    del llm
write_status("threads")
'''),

    md('## 6. Batch-1 latency and load time, Q8_0 vs BF16'),
    code('''
RESULTS["encode"] = {}
RESULTS["load_seconds"] = {}
for tag, fn in (("q8_0", "embeddinggemma-2-Q8_0.gguf"), ("bf16", "embeddinggemma-2-BF16.gguf")):
    t0 = time.perf_counter()
    llm = load(paths[fn], n_cpu)
    RESULTS["load_seconds"][tag] = time.perf_counter() - t0

    i = {"n": 0}
    def short_call(llm=llm, i=i):
        t = SHORT[i["n"] % len(SHORT)]; i["n"] += 1
        return embed_one(llm, PREFIX_QUERY + t)
    RESULTS["encode"][f"{tag}_short"] = timed(short_call, 20, warmup=3)

    j = {"n": 0}
    def long_call(llm=llm, j=j):
        t = LONG[j["n"] % len(LONG)]; j["n"] += 1
        return embed_one(llm, PREFIX_QUERY + t)
    RESULTS["encode"][f"{tag}_long"] = timed(long_call, 8, warmup=2)

    for k, v in RESULTS["encode"].items():
        if k.startswith(tag):
            print(f"{k:<12} mean {v['mean_ms']:8.1f} ms | p50 {v['p50_ms']:8.1f} | p95 {v['p95_ms']:8.1f}")
    print(f"{tag} load: {RESULTS['load_seconds'][tag]:.2f} s")
    if tag == "bf16":
        llm_bf16 = llm
    else:
        llm_q8 = llm
write_status("encode")
'''),

    md('''
## 7. Quality under Q8_0 — is it still the same model?

Banking77 zero-shot intent routing, the same task and the same prefixes as the PyTorch run that
gave **92.92%**. Data loaded from the canonical CSVs; label order taken from the upstream
dataset script.
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
O_q8 = np.stack([embed_one(llm_q8, PREFIX_DOC + o) for o in option_texts])
S_q8 = np.stack([embed_one(llm_q8, PREFIX_QUERY + t) for t in test_texts])
encode_s = time.perf_counter() - t0
RESULTS["banking77_throughput"] = {
    "texts": len(S_q8) + len(O_q8), "seconds": encode_s,
    "texts_per_second": (len(S_q8) + len(O_q8)) / encode_s}

sys.path.insert(0, "/kaggle/working/nanocore-s1")
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

    md('## 8. Drift: Q8_0 vs BF16'),
    code('''
sample = np.arange(0, len(test_texts), max(1, len(test_texts) // 150))[:150]
S_bf16 = np.stack([embed_one(llm_bf16, PREFIX_QUERY + test_texts[i]) for i in sample])
S_q8s = S_q8[sample]
cos = (S_bf16 * S_q8s).sum(axis=1)
RESULTS["drift"] = {"n": int(len(sample)), "mean_cosine": float(cos.mean()),
                    "min_cosine": float(cos.min()), "p05_cosine": float(np.percentile(cos, 5)),
                    "mean_abs_diff": float(np.abs(S_bf16 - S_q8s).mean())}
print(json.dumps(RESULTS["drift"], indent=2))
write_status("drift")
'''),

    md('## 9. Verdict against the PyTorch path'),
    code('''
q8_short = RESULTS["encode"]["q8_short"]["mean_ms"]
q8_long = RESULTS["encode"]["q8_long"]["mean_ms"]
PYTORCH_CPU_SHORT, PYTORCH_CPU_LONG = 363.5, 6595.0

RESULTS["verdict"] = {
    "q8_short_ms": q8_short, "pytorch_cpu_short_ms": PYTORCH_CPU_SHORT,
    "short_speedup": PYTORCH_CPU_SHORT / q8_short,
    "q8_long_ms": q8_long, "pytorch_cpu_long_ms": PYTORCH_CPU_LONG,
    "long_speedup": PYTORCH_CPU_LONG / q8_long,
    "q8_throughput_texts_per_s": RESULTS["banking77_throughput"]["texts_per_second"],
    "pytorch_cpu_texts_per_s": 1000.0 / 416.1,
    "throughput_speedup": RESULTS["banking77_throughput"]["texts_per_second"] / (1000.0 / 416.1),
    "quality_preserved": bool(RESULTS["banking77_quality_q8"]["accuracy_delta"] > -0.01),
    "thread_scaling_1_to_max": (RESULTS["threads"][f"n_threads=1"]["mean_ms"] /
                                RESULTS["threads"][f"n_threads={n_cpu}"]["mean_ms"]
                                if f"n_threads={n_cpu}" in RESULTS["threads"] else None),
}
print(json.dumps(RESULTS["verdict"], indent=2))

(WORK / "REPORT.md").write_text("# llama.cpp path\\n\\n" + json.dumps(RESULTS, indent=2, default=str))
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
    # the protocol test file too — the quality measurement uses the protocol, so it is
    # verified in-kernel rather than assumed
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
