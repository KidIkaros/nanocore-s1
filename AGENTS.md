# AGENTS.md — nanocore-s1 working notes

Guidance for agents and humans working in this repository.

## Compute boundary — where things run

| Work | Where |
|---|---|
| Unit suite, protocol tests, pure evaluation logic on cached arrays | local |
| Head training on cached embeddings (~100MB) | local |
| Anything that loads model weights (EG2, Laya) or encodes real text | **Kaggle** |
| Full multimodal inference | **Kaggle** |

Measured local/CPU throughput for planning (Kaggle 4-core Xeon, AVX2):
PyTorch fp32 ≈ **2.4 texts/s**, llama.cpp Q8_0 ≈ **9.9 texts/s**; batching gives
nothing on CPU. Encoding ~24k texts on CPU is ~2.7 h (PyTorch) / ~40 min (llama.cpp).

## Kaggle hygiene — one session limit, two rules

**Kaggle allows at most 2 concurrent GPU sessions per account, and queued sessions
count against the limit.** This has bitten us once already: a stuck kernel was
"retried" by re-pushing, each push stacked another queued session, both slots were
consumed, and every subsequent push failed with
`Maximum batch GPU session count of 2 reached` — a self-inflicted deadlock.

Rules:

1. **Never re-push a kernel that is already QUEUED or RUNNING.** Poll its status
   instead. If it must be replaced, delete it first (below) so its session is freed.
2. **Close out sessions when done.** Once a run's artifacts are pulled into
   `reports/runs/<run>/`, delete the throwaway kernel if it is not the canonical
   notebook — `kaggle kernels delete <owner>/<slug> -y`. Canonical notebooks (the
   ones we iterate on) stay; one-off kernels go.
3. There is **no CLI `stop`** — only `delete`. Queued/running sessions are cleared by
   deleting the kernel or by stopping it in the Kaggle UI.
4. **Quota is session wall-time, not compute time.** The GPU is attached for the whole
   session — weight downloads, pip installs, and compiles bill identically to inference.
   Minimize what happens inside a GPU session: ship prebuilt artifacts, mount models as
   datasets, keep notebooks single-purpose. Duplicate queued sessions each bill a full
   run — retry-pushing costs quota twice over (slots + hours).
5. **Interactive editor sessions are invisible to `kernels status`** — a notebook left
   open in the Kaggle editor with GPU on burns quota continuously. Check the UI's active
   sessions before assuming quota is safe.

## Kaggle gotchas found the hard way

- **`machine_shape` is mandatory** in `kernel-metadata.json`. With only
  `enable_gpu: true`, pushes default to P100 (sm_60) and modern PyTorch needs sm_70+.
  Use `"machine_shape": "NvidiaTeslaT4"`, and assert the GPU in the first cell.
- **The kernel title must slugify to the kernel id**, or the push fails with a bare 400.
- **Notebook outputs must be a single `artifacts.zip`** — `kaggle kernels output`
  fails on output subfolders.
- **Legacy no-namespace HF dataset ids now fail** (`clinc_oos` → `HfUriError`).
  Every data loader needs a fallback path (GitHub TSV/JSON); see
  `notebooks/s1_clinc150_oos/generate.py` and `notebooks/s1_goemotions_head/generate.py`.
- **`kaggle` lives in `.venv` without being on PATH** — call
  `/home/ikaaros/Coding/Gold/jev-stack/.venv/bin/kaggle`.

## Experiment conventions

- **One notebook per experiment family, iterating versions** — do not create a new
  notebook directory for each iteration. Kaggle versions natively; git history keeps
  the old generator. Per-run *evidence* still goes to `reports/runs/<run>/` so every
  number traces to an artifact.
- **Every kernel**: bundle the repo source, run `tests/test_decision_protocol.py` as a
  contract gate, write `run_status.json`, save per-example score matrices (`.npz`) so
  future evaluation re-runs need no encoding.
- **Three-way splits** for anything calibration-shaped: fit / calibrate / evaluate
  must be disjoint. Sharing data biases coverage optimistically.
- **Temperature is two things**: the value fitted for reported probabilities and the
  value used to build conformal sets are not interchangeable. Floors and per-scorer
  calibration are required — log-loss fitting otherwise sharpens to degeneracy
  (measured: T=0.02 → q̂=0.9975; head probs → q̂≈1.0).

## Verification

```bash
python -m pytest tests/ -q          # unit + protocol suite, no model loading
```

Kernels are syntax-checked cell-by-cell and their pure-logic paths dry-run on
synthetic arrays locally before any push.
