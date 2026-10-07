# ADR-0006: Anything touching model weights or real arrays executes on Kaggle, never locally

**Date**: 2026-10-07
**Status**: accepted (recorded after a second violation)
**Deciders**: project owner, agent

## Context

This project's host is small. In the predecessor repository the same mistake took
the machine down: loading a 421M-parameter model (~1.7 GB fp32 plus activations) on a
host reporting **1.6 GB available** drove load average to 78 on 12 cores, triggered a
swap storm, and the OS OOM-killed and restarted. That repository responded with a
memory guard (`requires_model(gb, what)`) whose rule is that a guard which fails open
is not a guard — unknown memory must skip, not pass.

On 2026-10-07 the same class of error happened again, in this repository, in a new
form: the agent executed a *notebook* locally as a "dry run" — importing torch,
loading a 19 MB embedding array, and fitting heads — after the owner had explicitly
required Kaggle execution. The host OOM-killed the session and the owner lost their
working state. The earlier guard did not help, because the failure was not a test
loading a model; it was arbitrary notebook code.

The GPU present on the host does not make this local work: the historical failure was
system RAM during weight loading, not VRAM.

## Decision

**All execution that imports torch, loads model weights, or operates on real data
arrays happens on Kaggle.** Local work is limited to: reading and writing files,
static checks (`ast.parse`, JSON validation, glob and path assertions), and `git`.

A notebook is validated by pushing a **reduced kernel** to Kaggle — a cheap fast-fail
run — not by executing it locally. Notebook generators are versioned in the repository
rather than in a temporary directory, so they survive a host restart.

| Work | Where |
|---|---|
| Reading, writing, editing, static checks, git | local |
| Notebook generation | local (text only) |
| Anything importing torch/numpy over real arrays | **Kaggle** |
| Encoder loading, embedding, head or composer training | **Kaggle** |
| Any number that appears in a results table | **Kaggle** |

## Alternatives Considered

### Alternative 1: Extend the memory guard to cover notebook execution
- **Pros**: keeps a single local workflow; reuses existing machinery.
- **Cons**: a guard estimates *declared* memory, and arbitrary notebook code declares
  nothing. The last failure would not have been caught by any such guard.
- **Why not**: it guards the wrong abstraction. A guard on "may I load this model"
  does not constrain "may I execute this notebook".

### Alternative 2: Run locally with reduced data and epochs
- **Pros**: fastest iteration; no upload or queue latency.
- **Cons**: still imports torch and still allocates; the failure mode is a spike, not
  a function of epoch count. Also validates a different program than the one that
  will be run for real.
- **Why not**: this is precisely the rationalisation that caused the outage.

### Alternative 3: Use the local GPU
- **Pros**: the host has one; faster than Kaggle CPU.
- **Cons**: irrelevant — the historical OOM was system RAM during weight loading, and
  the recent failure was RAM from an embedding array, not VRAM.
- **Why not**: misdiagnoses the constraint.

## Consequences

### Positive
- The host cannot be taken down by this project again.
- Kaggle runs are the artifact of record, so results are reproducible from the kernel
  and its metadata rather than from an untracked local session.
- Kaggle's free tier is sufficient; the quota is published and runs are scriptable.
  Cached-embedding analysis can run on a CPU-only kernel and spend **no GPU quota** —
  the calibration run used this and cost nothing.

### Negative
- Iteration is slower: a kernel push, a queue wait, and a run.
- Notebook bugs cost a full run, which makes pre-push static verification load-bearing.

### Risks
- **Silent local execution creeping back.** Mitigation: this ADR is referenced from
  the architecture map (§0) and from the compute-boundary section of the build plan,
  and each notebook states its execution environment in its first cell.
- **Lost results on host restart.** /tmp is not durable and was wiped mid-project,
  losing a notebook generator and a run's artifacts. Mitigations now in place:
  generators versioned in the repo, and every kernel stage writes accumulated results
  to disk so a failed run still yields everything it completed.
- **Notebooks fail late and lose everything.** Mitigation: incremental result
  persistence, as above, plus reduced-kernel fast-fail for validation.
