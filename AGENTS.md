# AGENTS.md — nanocore-s1 working notes

Guidance for agents and humans working in this repository.

> **Read `docs/ROADMAP.md` first.** It is the single source of truth for what is being
> built and what "done" means. If a proposed action is not in the roadmap, add it there
> with acceptance criteria before doing it. The stop-doing list in §6 is binding.

## Compute boundary — where things run

**Kaggle is the workspace. The local machine edits files, reads files, runs static
checks, and does git operations — nothing else.** This is the roadmap's R1 and it is
binding. The host has frozen once under a model load already; do not re-test it.

| Work | Where |
|---|---|
| Code/document edits, file reads, `git` | local |
| Static checks (pytest unit suite, syntax checks, no weights) | local |
| **Anything that loads model weights or encodes real text** | **Kaggle only** |
| Head/training on real embeddings, full inference | **Kaggle only** |
| **Installing or running model runtimes (llama.cpp, GGUF inference)** | **Kaggle only, until the on-device phase** |
| Multimodal inference | **Kaggle only** |

Never `pip install` or build a model runtime on this host. Never run the CLI's encoder
path locally. The on-device phase (roadmap Phase 8) is the only exception, and the
owner triggers it explicitly.

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
- **OAuth tokens expire mid-session.** `~/.kaggle/credentials.json` carries an
  `access_token` with ~1h TTL and the CLI does not refresh it — calls start
  failing with `Permission 'kernels.get' was denied` or a bare auth prompt.
  `kaggle auth print-access-token` mints a fresh token; pass it as
  `KAGGLE_API_TOKEN` for subsequent calls (it does not update the stored file).

## Experiment conventions

- **Push only on the owner's word.** A kernel run costs ~45 min of quota; pushing
  and then immediately finding more fixes wastes the run. Before any push the
  working tree must be clean of known fixes: run the clean-code and
  clean-architecture review pass *after every fix*, not after the push. When
  the owner says push, the code that goes up is the code that was reviewed.
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
