# OLC6 sealed-model-container — prototype

Roadmap component 44 · ADR-0015's distribution layer · spike, not production code.

## What this demos

A NanoCore bundle packed as **slots in a single sealed OLC6 vault** — the physical
container for the world-model-container idea. `./demo.sh` runs the whole loop on the
real `origin-container` binary (owner's sandbox project) against the real v24 bundle:

```
stage/ = the model's "body"
  manifest.json scorer.pt gate.json      ← decision ability (real, qualified v24)
  glial_state.json                        ← regulator checkpoint (ADR-0014)
  transition_head.json                    ← z'=f(z,a) slot (ADR-0015, reserved)
  memory_index.json                       ← trajectory spine (component 42)
  provenance.json MODEL_CARD.md           ← identity + documentation

one vault file = one model generation
```

## Primitives exercised, and what each proves for a stateful model

| OLC6 verb | What the demo shows | Why a stateful model needs it |
|---|---|---|
| `init` | 8 slots → one 612KB sealed file | single artifact, atomic, encrypted at rest |
| `list` | slot dir without decrypting bodies | inventory without materializing weights |
| `verify` | all field tags authenticate | integrity on arrival — model fingerprinting |
| `snapshot` | `nanocore.gen-0001.olc` sidecar | generation retention = the rollback path |
| `write` (glial slot) | gen-2 seal; vault **shrank 12 bytes** | **delta-priced updates**: overnight glial consolidation ships only the bytes that changed — not the whole bundle |
| `sync push/pull` | lineage+generation+digest identity check; divergence refused loudly | on-device delivery with version safety |
| `export` (3 slots) | decision-only `.oexp` under its own passphrase | capability-limited projection for a constrained device — no glial/memory slots leave |
| `triage` | header parses with no passphrase | what an untrusted party can see: format, generation, slot count — never names or bodies |
| `recover`/`crash-test` | dual table copies + monotonic revision | persistent regulator/memory state survives a crash mid-seal |

## The mapping (the point of the exercise)

```
OLC6 seal      ≈ model release      — monotonic generations, tamper-evident
slot           ≈ ability artifact   — scorer/head/state each get a stable uid
delta seal     ≈ consolidation push — sleep-phase writes → byte-priced OTA update
sync           ≈ device delivery    — sealed generations between machines
export         ≈ capability         — ship a decision-only model to a device
                 projection           without shipping the regulator it learned from
snapshot       ≈ rollback           — earn-the-slot adoption needs a working prior
```

And the deep correspondence: OLC6's invariant is *projections consume seals; they
never redefine them* — the container analogue of *abilities read `z`; none redefine
the encoder*.

## Honest bounds

- **Distribution layer only.** This is the physical artifact container — it does
  not touch the latent state space `z` the world-model claim lives in. They
  compose; they are not the same thing.
- **Reference posture.** `origin-container` is a Rust crate, NanoCore is Python —
  production integration is a PyO3/FFI or format-spec-port question, deferred to
  Phase 8 (on-device delivery is where it earns). Owner posture I-1:
  reference-only, recreate-on-need.
- **Key management is unpriced.** A sealed bundle needs a passphrase wherever it
  runs; on-device key storage is a design question this spike deliberately skips.
- **Demo passphrase is a placeholder** (`ORIGIN_VAULT_PW=prototype-demo-only…`),
  never a real secret.

## Run it

```sh
./demo.sh     # uses the sandbox release binary; override with ORIGIN_CONTAINER_BIN
```
