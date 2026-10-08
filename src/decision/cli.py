"""Command-line interface — a locally runnable decision model.

    python -m src.decision.cli decide "i need to cancel my flight" \
        --options "cancel,book flight,balance,transfer" \
        --model models/embeddinggemma-300M-Q8_0.gguf \
        --bundle models/clinc_bundle

    # batch mode — one input per line, single encoder load:
    python -m src.decision.cli decide - --options "..." --inputs demos.txt --json

Without ``--bundle`` the CLI is ungated: it reports probabilities only.
With a bundle it runs the full pipeline — calibrated probabilities, conformal
prediction set, and an action (answer / clarify / escalate).

``--backend st`` uses the sentence-transformers reference encoder (Kaggle/dev);
``--backend llamacpp`` uses a GGUF file (the on-device deployment path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "models/embeddinggemma-300M-Q8_0.gguf"


def _build(args):
    from src.decision.cache import DecisionCache
    from src.decision.model import DecisionModel

    if args.backend == "st":
        from src.decision.encoder import StateEncoder
        encoder = StateEncoder(modalities=("text",), device=args.device or None)
    else:
        from src.decision.backends import LlamaCppEncoder
        if not Path(args.model).exists():
            sys.exit(f"model file not found: {args.model}\n"
                     f"download it with:\n"
                     f"  python -c \"from huggingface_hub import hf_hub_download as d;"
                     f"print(d('ggml-org/embeddinggemma-300m-GGUF','embeddinggemma-300M-Q8_0.gguf',"
                     f"local_dir='models'))\"")
        encoder = LlamaCppEncoder(args.model, max_tokens=args.max_tokens)

    cache = None if args.no_cache else DecisionCache(args.cache)
    if args.bundle:
        return DecisionModel.load(args.bundle, encoder=encoder, cache=cache)
    return DecisionModel(encoder=encoder, cache=cache)


def _texts(args):
    if args.inputs:
        return [l.strip() for l in Path(args.inputs).read_text().splitlines() if l.strip()]
    return [args.text]


def cmd_decide(args):
    from src.decision.schema import Question

    model = _build(args)
    options = [o.strip() for o in args.options.split(",") if o.strip()]
    if not options:
        sys.exit("--options must list at least one option")
    q = Question(qtype=args.qtype, options=options,
                 scale=[float(x) for x in args.scale.split(",")] if args.scale else None)

    rows = []
    for text in _texts(args):
        rows.append((text, model.decide(text, q, policy=args.policy)))

    if args.json:
        out = [{"input": t, "qtype": p.qtype, "action": p.action,
                "top_prob": p.answer_confidence, "prediction_set": p.prediction_set,
                "probabilities": p.probabilities, "alpha": p.alpha} for t, p in rows]
        print(json.dumps(out if len(out) > 1 else out[0], indent=1))
        return

    for text, p in rows:
        ranked = sorted(p.probabilities.items(), key=lambda kv: -kv[1])[:5]
        print(f'input   : {text}')
        print(f'action  : {p.action}')
        print(f'top prob: {p.answer_confidence:.4f}')
        if p.qtype == "score" and p.score is not None:
            print(f'expected: {p.score:.3f}')
        if p.prediction_set:
            print(f'set     : {", ".join(p.prediction_set[:8])}'
                  + (" ..." if len(p.prediction_set) > 8 else ""))
        print("top-5   : " + "  ".join(f"{l}={v:.3f}" for l, v in ranked))
        print("-" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nanocore", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide", help="one typed decision (per line with --inputs)")
    d.add_argument("text", nargs="?", default="-", help="input text (ignored with --inputs)")
    d.add_argument("--inputs", help="file with one input text per line")
    d.add_argument("--options", required=True, help="comma-separated option labels")
    d.add_argument("--qtype", default="choice", choices=["choice", "score", "noul"])
    d.add_argument("--scale", help="numeric scale for score questions")
    d.add_argument("--policy", default=None, choices=["full", "escalate", "answer"])
    d.add_argument("--backend", default="st", choices=["st", "llamacpp"])
    d.add_argument("--device", help="torch device for --backend st (default: auto)")
    d.add_argument("--model", default=DEFAULT_MODEL, help="GGUF path for --backend llamacpp")
    d.add_argument("--bundle", help="calibrated bundle dir (enables the gate)")
    d.add_argument("--cache", default=".nanocore-cache")
    d.add_argument("--no-cache", action="store_true")
    d.add_argument("--max-tokens", type=int, default=512)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_decide)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
