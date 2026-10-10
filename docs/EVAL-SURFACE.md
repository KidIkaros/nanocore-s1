# Evaluation surface — questions, not legs

> Status: **design** (2026-10-10). Reorganizes the Kaggle kernel family around the
> questions a run should answer about NanoCore, instead of the mechanisms a leg
> exercises. The current notebook list (17 dirs) is mostly *historical experiment
> residue* — one-off questions from development, later folded into `s1_verify`.

## The problem being fixed

Every push of `s1_verify` **rebuilds the model then tests it**: download datasets,
encode corpora, fit heads, calibrate the gate, then run ~20 legs against the thing it
just built. Two consequences:

1. **It's expensive where it doesn't need to be.** Most legs could run against the
   *last qualified bundle* — the artifact, not a fresh build. Nothing persists between
   runs, so every run pays full encode+fit cost before asking anything.
2. **It asks qualification questions only.** Every leg is one-shot scoring on benchmark
   rows. It has never seen a conversation, a typo, a frustrated user, gradual drift,
   or an image used as conversational state — i.e., it cannot answer *"how does it
   respond to human stimulus?"*, which is the actual production-readiness question.

## The question → kernel matrix

| # | Question | Kernel | Builds the model? | Cadence |
|---|----------|--------|-------------------|---------|
| **Q1** | Does the *pipeline* produce a qualified bundle? — calibration, gates G1–G4, baselines, roundtrip, Jev anchor, ops machinery | `s1_verify` | **Yes — that is the point.** A verification run builds so it can verify the build | Rare — on pipeline/architecture change only (encoder version, scorer, gate logic, calibration machinery) |
| **Q2** | How does it *behave* with humans? — real-world stimulus, action-appropriateness, gradual drift, multi-turn context, multimodal state | **`s1_stimulus`** (component 45) | **No — mounts Q1's saved bundle, frozen** | Often — every meaningful change; it's the cheap run |
| **Q3** | Does it fit and run on device? — llama.cpp/GGUF, latency, memory, quantization | `s1_llamacpp*` (dormant) | No — converts Q1's bundle | Phase 8 only |
| **Q4** | Does the code work? — torch-tier unit + protocol suite, contract gate | `s1_tests` | No | When torch-tier code changes |

Two questions exist that no kernel can ask yet — the gap is explicit:

- **Q5 — Is `f` learnable?** (transition head, component 43): needs the trajectory
  corpus that only Q2 produces. Blocked by construction.
- **Q6 — Does it converse?** (the conversation ability): needs the composer factory
  (component 38). Q2's multi-turn legs degrade to history-as-text until then.

## The dependency structure

```
s1_verify ──(rare)──► qualified bundle saved as a Kaggle DATASET artifact
  build + qualify           │
                            ▼  mounted, frozen — never rebuilt downstream
s1_stimulus ──(often)──► behavior report + trajectory corpus
  test the artifact         │
                            ▼
                    component-42 corpus → feeds Q5 feasibility probe,
                                          glial retraining (41),
                                          outcome-joined metrics
```

**The bundle-as-dataset is the load-bearing change.** `s1_verify` must *save* the
qualified bundle (manifest + scorer + gate + metadata) as a versioned Kaggle dataset
artifact — then every other kernel mounts it read-only. "Don't rebuild the model
every time" is a literal requirement, not a preference: the behavior questions must
be answered about the *shipped artifact*, or the numbers don't transfer.

## Q2 — `s1_stimulus` design sketch

Benchmark-driven, not hand-scripted (datasets carry *natural labels* for the action
question — hand-written stimulus tests our imagination of humans; corpora test humans):

| Behavior under test | Dataset source | Natural label |
|---|---|---|
| Real human input distribution | `lmsys/lmsys-chat-1m`, `allenai/WildChat` | none — measures *our* action distribution on real stimulus (drift/competence mirror) |
| Should-have-abstained | `squad_v2` unanswerable split | `is_impossible` → expect escalate/abstain |
| Should-have-clarified | `sewon/ambig_qa`, ClariQ | ambiguity annotation → expect clarify |
| OOS intent | `clinc_oos` oos class (already used) | oos label → expect escalate/abstain |
| Multi-turn context | `multi_woz_v22`, `sgd`, `quac` | dialogue state → contextual correctness (degraded until 38) |
| Response selection | `mutual`, `udc` | correct continuation → typed decision over conversation state |
| Gradual/temporal drift | `wilds` (amazon, civil) | domain label → braking curve vs ramp |
| Adversarial robustness | `anli`, `advglue` | gold label → escalation correctness under attack |
| Image+text decisions | `Mind2Web`, `android_in_the_wild`, `vqa` | action/answer label → multimodal-state decisions |
| Emotional register | `go_emotions`, `empathetic_dialogues` | robustness axis, not a correctness gate |

*(IDs to verify when wiring — several have canonical re-uploads; license review
required before publishing "evaluated on" claims — LMSYS is research-only.)*

**Scoring changes too.** `s1_verify` scores *accuracy/calibration*; `s1_stimulus`
scores **action-appropriateness**: did it clarify when the datum was ambiguous,
escalate when unanswerable, answer when it should, brake at the right point on a
drift ramp. Different metric, different gate — this is the production-readiness
evidence the qualification number isn't.

**And it produces the corpus.** Every stream runs through the shipped `serve` path
with stream linkage on — each benchmark emits real trajectory rows
(`stream_id`/`prev_id`/`outcome`), so the first component-42 corpus falls out of the
behavior run as a side effect. Q2 is therefore the unblocker for Q5 *and* the
production evidence, in one run.

## Notebook disposition

**Active (4):** `s1_verify` (Q1), `s1_stimulus` (Q2, new), `s1_llamacpp`+`server`
(Q3, dormant until Phase 8), `s1_tests` (Q4).

**Retire to archive (≈13):** `s1_adapt`, `s1_baseline`, `s1_calibration`,
`s1_clinc150_oos`, `s1_dispatch`, `s1_goemotions`, `s1_goemotions_head`,
`s1_latency`, `s1_latency_gpu`, `s1_llamacpp_server` (fold into Q3),
`s1_policy`, `s1_prompt_ablation`, `s1_sharpening`, `nanocore_audit`.
Git history keeps the generators; Kaggle-side deletions free the account and the
2-session limit. Their questions are either answered (recorded in `reports/runs/`)
or live on as `s1_verify` legs.

## Honest gaps in this design

- **Q2 can't fully run today**: no qualified bundle has ever been saved as a
  mountable dataset (s1_verify saves `.npz` evidence, not a deployable artifact), and
  multi-turn legs degrade until component 38. First sequence is necessarily
  Q1(with bundle-save added) → Q2.
- **"Action-appropriateness" needs rubric discipline**: natural labels exist for
  abstain/clarify cases; for real-chat corpora there is no gold — we measure
  distribution + agreement, not correctness. The doc must not imply a gold where
  none exists.
- **Retiring notebooks is irreversible on Kaggle's side** — confirm each one's
  latest run is recorded in `reports/runs/` before deleting.
