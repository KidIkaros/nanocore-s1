#!/usr/bin/env bash
# OLC6 sealed-model-container prototype (roadmap component 44 / ADR-0015 distribution layer)
#
# Packs a real NanoCore bundle (reports/runs/v24/ops/retrain1/bundle) plus the
# future ability slots (glial state, transition head, memory, model card) into a
# single sealed OLC6 vault, then demos the primitives that matter for a
# stateful, learning, on-device model:
#
#   init      one sealed file = one model generation
#   list      slot directory without decrypting bodies
#   verify    full-field integrity on the sealed artifact
#   snapshot  generation retention -> the rollback path
#   write     a "sleep consolidation" touches ONE slot -> seal prices edited bytes
#   sync      deliver the sealed generation to a second copy (device), verify there
#   export    capability-limited projection (decision-only) for a constrained device
#   inspect   what an untrusted party sees WITHOUT the passphrase
#
# Usage:
#   ORIGIN_CONTAINER_BIN=/path/to/origin-container ./demo.sh
#   (defaults to the sandbox release build)
set -euo pipefail

OC="${ORIGIN_CONTAINER_BIN:-/home/ikaaros/Coding/Gold/SANDBOX/origin-container/target/release/origin-container}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUNDLE="${BUNDLE:-$REPO/reports/runs/v24/ops/retrain1/bundle}"
WORK="$(mktemp -d /tmp/olc-wm-XXXXXX)"
export ORIGIN_VAULT_PW='prototype-demo-only-not-a-secret'
export ORIGIN_EXPORT_PW='export-demo-only-not-a-secret'

step() { printf '\n=== %s ===\n' "$*"; }

# --- stage the container contents -------------------------------------------
step "Stage NanoCore container contents (real v24 bundle + future ability slots)"
mkdir -p "$WORK/stage"
cp "$BUNDLE"/manifest.json "$BUNDLE"/scorer.pt "$BUNDLE"/gate.json "$WORK/stage/"
# future slots — placeholders so the slot map shows the full container shape
cat > "$WORK/stage/glial_state.json" <<'EOF'
{"slot": "glial", "version": 0, "note": "regulator state checkpoint — written by sleep consolidation"}
EOF
cat > "$WORK/stage/transition_head.json" <<'EOF'
{"slot": "transition", "version": 0, "note": "z' = f(z,a) — ADR-0015, proposed; empty until trained"}
EOF
cat > "$WORK/stage/memory_index.json" <<'EOF'
{"slot": "memory", "episodes": [], "note": "trajectory log spine — component 42"}
EOF
cat > "$WORK/stage/provenance.json" <<EOF
{"bundle_source": "$(basename "$BUNDLE")", "sealed": "$(date -u +%FT%TZ)", "origin": "olc-container prototype"}
EOF
cat > "$WORK/stage/MODEL_CARD.md" <<'EOF'
# NanoCore container (prototype) — decision ability qualified (v24); glial/transition/memory slots reserved.
EOF
ls -l "$WORK/stage"

# --- seal generation 1 -------------------------------------------------------
step "init -> seal generation 1 (one file = one model generation)"
"$OC" init "$WORK/nanocore.olc" "$WORK/stage/"*.json "$WORK/stage/"*.pt "$WORK/stage/MODEL_CARD.md" >/dev/null
"$OC" list "$WORK/nanocore.olc"
VAULT_SIZE_1=$(stat -c %s "$WORK/nanocore.olc")
echo "vault size gen1: $VAULT_SIZE_1 bytes"

step "verify sealed artifact (integrity, all fields)"
"$OC" verify "$WORK/nanocore.olc" | tail -3

step "snapshot gen1 -> rollback point"
"$OC" snapshot "$WORK/nanocore.olc" >/dev/null
ls "$WORK"/*.gen-*.olc

# --- the delta demo: sleep consolidation writes ONE slot ---------------------
step "sleep consolidation: rewrite glial_state.json only, seal"
cat > "$WORK/glial_state.json" <<'EOF'
{"slot": "glial", "version": 1, "competence_field": "consolidated overnight", "regions": 12}
EOF
"$OC" write "$WORK/nanocore.olc" --slot glial_state.json --file "$WORK/glial_state.json" >/dev/null
VAULT_SIZE_2=$(stat -c %s "$WORK/nanocore.olc")
echo "vault size gen2: $VAULT_SIZE_2 bytes (delta $((VAULT_SIZE_2 - VAULT_SIZE_1)) — priced by edited bytes, not vault size)"
"$OC" inspect "$WORK/nanocore.olc" | grep -i -E "generation|revision" || true

# --- delivery: sync to a "device" copy ---------------------------------------
step "sync push -> device copy, verify on arrival"
cp "$WORK/nanocore.olc" "$WORK/device.olc"
"$OC" sync push "$WORK/nanocore.olc" "$WORK/device.olc"
"$OC" verify "$WORK/device.olc" | tail -3

# --- capability projection: decision-only export -----------------------------
step "export decision-only projection (no glial/transition/memory slots)"
"$OC" export "$WORK/nanocore.olc" --slot manifest.json --slot scorer.pt --slot gate.json \
    --out "$WORK/decision-only.oexp"

# --- untrusted view -----------------------------------------------------------
step "inspect WITHOUT passphrase semantics (triage: header only)"
env -u ORIGIN_VAULT_PW "$OC" triage "$WORK/nanocore.olc" || true

# --- rollback -----------------------------------------------------------------
step "rollback: gen1 sidecar is itself a sealed vault"
SNAP="$(ls "$WORK"/*.gen-*.olc | head -1)"
"$OC" verify "$SNAP" | tail -2
echo "rollback artifact: $SNAP"

step "DONE — artifacts in $WORK"
echo "  vault:        $WORK/nanocore.olc"
echo "  device copy:  $WORK/device.olc"
echo "  projection:   $WORK/decision-only.oexp"
echo "  snapshot:     $SNAP"
