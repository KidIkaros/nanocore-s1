"""Command-line interface — a locally runnable decision model.

Three commands, in the order a user meets them:

    # 1. labeled data -> a calibrated bundle
    python -m src.decision.cli adapt --data tickets.csv \
        --text-col text --label-col intent --out models/tickets

    # 2. one typed decision, using that bundle
    python -m src.decision.cli decide "i need to cancel my flight" \
        --options "cancel,book flight,balance,transfer" \
        --model models/embeddinggemma-300M-Q8_0.gguf \
        --bundle models/tickets/bundle

    # 3. the same thing over HTTP
    python -m src.decision.cli serve --bundle models/tickets/bundle

    # and, on the target device: does the GGUF agree with the PyTorch reference?
    python -m src.decision.cli parity --inputs demos.txt --options "..." \
        --bundle models/tickets/bundle --backend llamacpp --model models/x.gguf

Batch decide: one input per line, single encoder load —
``decide - --options "..." --inputs demos.txt --json``.

Without ``--bundle`` the CLI is ungated: it reports probabilities only.
With a bundle it runs the full pipeline — calibrated probabilities, conformal
prediction set, and an action (answer / clarify / escalate).

``--backend st`` uses the sentence-transformers reference encoder (Kaggle/dev);
``--backend llamacpp`` uses a GGUF file (the on-device deployment path);
``--backend stub`` loads no weights at all, for tests and dry runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "models/embeddinggemma-300M-Q8_0.gguf"


def _backend_args() -> argparse.ArgumentParser:
    """Encoder-selection flags, declared once for every command that needs one.

    Three commands build an encoder now, and three copies of four flags is how a
    default drifts out of step between them.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--backend", default="st", choices=["st", "llamacpp", "stub"])
    p.add_argument("--device", help="torch device for --backend st (default: auto)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="GGUF path for --backend llamacpp")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--ignore-memory", action="store_true",
                   help="load weights even if free RAM looks insufficient")
    return p


def _model_args() -> argparse.ArgumentParser:
    """``_backend_args`` plus caching, for commands that build a ``DecisionModel``.

    Separate from the encoder-only parent because ``parity`` builds encoders and
    never a model — carrying cache flags it cannot use is how a flag ends up
    documented and ignored.
    """
    p = argparse.ArgumentParser(add_help=False, parents=[_backend_args()])
    p.add_argument("--cache", default=".nanocore-cache")
    p.add_argument("--no-cache", action="store_true")
    return p


def _require_memory(backend: str, model: str, ignore: bool) -> None:
    """Refuse to load weights without RAM to spare. Exits; returns no verdict.

    Called from ``_encoder`` rather than by each command so a new command cannot
    forget it — a guard that has to be remembered is not a guard. It must also
    run *before* the loader starts, which is the only moment it can help: once
    the weights are being read, the RAM is already committed.
    """
    from src.decision.guard import ENCODER_GB, check_memory, model_gb
    need = ENCODER_GB["text"] if backend == "st" else model_gb(model)
    check = check_memory(need)
    if check.ok:
        if check.available_gb is None:
            print(f"note: {check.reason}", file=sys.stderr)
    elif not ignore:
        sys.exit(f"refusing to load weights: {check.reason}")
    else:
        print(f"WARNING: --ignore-memory set; {check.reason}", file=sys.stderr)


def _encoder(backend: str, model: str = DEFAULT_MODEL, device=None,
             max_tokens: int = 512, ignore_memory: bool = False):
    """Build the encoder asked for. ``stub`` loads no weights, so it has no gate."""
    if backend == "stub":
        from src.decision.backends import StubEncoder
        return StubEncoder()                        # dry-runs/tests, no weights

    if backend != "st" and not Path(model).exists():
        sys.exit(f"model file not found: {model}\n"
                 f"download it with:\n"
                 f"  python -c \"from huggingface_hub import hf_hub_download as d;"
                 f"print(d('ggml-org/embeddinggemma-300m-GGUF',"
                 f"'embeddinggemma-300M-Q8_0.gguf',local_dir='models'))\"")
    _require_memory(backend, model, ignore_memory)

    if backend == "st":
        from src.decision.encoder import StateEncoder
        return StateEncoder(modalities=("text",), device=device or None)
    from src.decision.backends import LlamaCppEncoder
    try:
        return LlamaCppEncoder(model, max_tokens=max_tokens)
    except ImportError as e:
        # The CLI is the boundary a user meets: a missing optional dependency is
        # an instruction ("pip install ..."), not a traceback.
        sys.exit(str(e))


def _build(args):
    from src.decision.cache import DecisionCache
    from src.decision.model import DecisionModel

    encoder = _encoder(args.backend, args.model, args.device, args.max_tokens,
                       args.ignore_memory)
    cache = None if args.no_cache else DecisionCache(args.cache)
    if args.bundle:
        return DecisionModel.load(args.bundle, encoder=encoder, cache=cache)
    return DecisionModel(encoder=encoder, cache=cache)


def _texts(args):
    if args.inputs:
        return [l.strip() for l in Path(args.inputs).read_text().splitlines() if l.strip()]
    return [args.text]


def _split_options(spec: str) -> list:
    """``"a, b ,c"`` → ``["a", "b", "c"]``. One parser for every command."""
    return [o.strip() for o in spec.split(",") if o.strip()]


def cmd_decide(args):
    from src.decision.schema import Question

    model = _build(args)
    options = _split_options(args.options)
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


def cmd_serve(args):
    model = _build(args)
    from src.decision.serve import serve

    model_id = args.bundle or args.model
    httpd = serve(model, host=args.host, port=args.port,
                  log_path=args.log, model_id=str(model_id),
                  policy=args.policy)
    print(f"serving on http://{args.host}:{args.port} "
          f"(model_id={model_id}, log={args.log})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def cmd_refusals(args):
    """Run the refusal battery — what this layer must decline, and must not.

    *Decision abstentions*, not content moderation: the model carries no content
    policy, so where the refusal boundary falls is a question about your risk
    posture. It is declared as costs (``--cfp/--cfn/--cr``) and the boundary
    follows — "assert when p is high, discard when it is low, review when it is
    intermediate" (Barbosa, arXiv:2609.28940).

    ``--inputs`` supplies real texts for the invariants to run over; ``--cases``
    adds your own should-not-do rules, which this layer will not invent for you.
    """
    from src.decision.refusals import (LossMatrix, battery_is_meaningful,
                                       battery_report, load_policy_cases,
                                       run_policy_cases)
    from src.decision.schema import Question

    model = _build(args)
    options = _split_options(args.options)
    loss = LossMatrix(cfp=args.cfp, cfn=args.cfn, cr=args.cr)
    texts = _texts(args) if (args.inputs or args.text != "-") else []

    rows = []
    for text in texts:
        pred = model.decide(text, Question(qtype="choice", options=options))
        rows.append({"top_prob": pred.answer_confidence, "max_score": pred.max_score,
                     "action": pred.action})

    report = battery_report(model, options, rows, sample_texts=texts,
                            gate=model.gate, loss=loss)
    if args.cases:
        report["policy"] = run_policy_cases(model, load_policy_cases(args.cases), options)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    th = report["loss"]
    print(f"cost posture   cfp={th['cfp']} cfn={th['cfn']} cr={th['cr']}  ->  "
          f"discard below p={th['discard_below']:.3f}, "
          f"assert above p={th['assert_above']:.3f}, review between")
    print(f"reasons ({len(report['reasons'])}):")
    for rid, v in report["reasons"].items():
        print(f"  {v['status']:10} {rid:26} [{v['family']:13}] via {v['verified_by']}")
    print("invariants:")
    for i in report["invariants"]:
        print(f"  {i['status']:10} {i['id']:3} {i['statement'][:58]}")
        if i["status"] in ("fail", "error"):
            print(f"             {i['detail'][:96]}")
    print(f"probes         unsafe answers {report['unsafe_answers']} "
          f"| over-refusals {report['over_refusals'] or 'none'}")
    print(f"meaningful     {battery_is_meaningful(report)} "
          f"(False means the battery was passed by asserting nothing)")
    if report.get("policy") is not None:
        bad = [p["case"] for p in report["policy"] if not p["pass"]]
        print(f"policy cases   {len(report['policy'])} run, "
              f"{'all declined' if not bad else f'NOT declined: {bad}'}")


def cmd_parity(args):
    """Compare the reference backend against the target one on the same inputs.

    Phase 8's acceptance run: does the on-device path (a GGUF) agree with the
    PyTorch reference closely enough to swap them? Both encoders are fed the
    *same* texts and the *same* bundle, so any difference is the backend's.
    """
    import time as _time

    from src.decision.encoder import to_numpy
    from src.decision.model import DecisionModel
    from src.decision.parity import (ParityTolerance, cosine_agreement,
                                     decision_agreement, latency_summary,
                                     parity_report)
    from src.decision.schema import Question

    texts = _texts(args)
    question = Question(qtype="choice", options=_split_options(args.options))
    tol = ParityTolerance(min_cosine=args.min_cosine,
                          min_action_agreement=args.min_action_agreement)

    encoders, preds, latencies = {}, {}, {}
    for role, backend in (("reference", args.reference), ("target", args.backend)):
        enc = _encoder(backend, args.model, args.device, args.max_tokens,
                       args.ignore_memory)
        encoders[role] = enc
        model = (DecisionModel.load(args.bundle, encoder=enc) if args.bundle
                 else DecisionModel(encoder=enc))
        out, times = [], []
        for text in texts:
            t0 = _time.perf_counter()
            out.append(model.decide(text, question))
            times.append((_time.perf_counter() - t0) * 1000)
        preds[role], latencies[role] = out, times
        print(f"{role:9} {backend:9} {len(texts)} decisions, "
              f"p50 {np.percentile(times, 50):.1f} ms")

    A = to_numpy(encoders["reference"].encode(texts, prompt_name="Classification",
                                              batch_size=args.batch))
    B = to_numpy(encoders["target"].encode(texts, prompt_name="Classification",
                                           batch_size=args.batch))
    rep = parity_report(cosine_agreement(A, B),
                        decision_agreement(preds["reference"], preds["target"]),
                        latency_summary(latencies["reference"]),
                        latency_summary(latencies["target"]), tol)

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        _print_parity_report(rep, tol)


def _print_parity_report(rep: dict, tol) -> None:
    """Cosine first: it is the metric that catches a shift argmax would hide."""
    c, d = rep["cosine"], rep["decisions"]
    print(f"cosine    mean {c['mean']:.4f}  min {c['min']:.4f}  p05 {c['p05']:.4f}  "
          f"(tolerance {tol.min_cosine})  -> {'OK' if rep['cosine_ok'] else 'FAIL'}")
    print(f"decisions action agreement {d['action_agreement']:.3f}  "
          f"set Jaccard {d['mean_set_jaccard']:.3f} "
          f"(tolerance {tol.min_action_agreement})  "
          f"-> {'OK' if rep['actions_ok'] else 'FAIL'}")
    lat = rep.get("latency") or {}
    ref, tgt = lat.get("reference") or {}, lat.get("target") or {}
    print(f"latency   reference p50 {ref.get('p50', float('nan')):.1f} ms | "
          f"target p50 {tgt.get('p50', float('nan')):.1f} ms | "
          f"ratio x{lat.get('p50_ratio', float('nan')):.2f}")
    print(f"parity    {'OK' if rep['parity_ok'] else 'FAIL'}")


def _print_adapt_report(rep: dict, bundle_dir) -> None:
    """What a user needs to judge the bundle they just made.

    Headroom first, because it says whether a trained head was the right call at
    all (ADR-0011): on a saturated task the zero-shot baseline already resolves
    the problem and no head can be justified.
    """
    test = rep["test"]
    print(f"data      {rep['n_total']} examples, {rep['n_classes']} classes "
          f"(fit {rep['splits']['fit']} / cal {rep['splits']['cal']} / "
          f"test {rep['splits']['test']})")
    print(f"headroom  zero-shot cosine already gets "
          f"{rep['headroom']['zeroshot_test_acc']:.3f} — below 0.85 there is "
          f"room for a head to help (ADR-0011)")
    print(f"head      {rep['head']['kind']}, "
          f"{rep['head'].get('epochs_run', '?')} epochs")
    print(f"test      accuracy {test['accuracy']:.3f} | Brier {test['brier']:.3f} "
          f"| coverage {test['conformal_coverage']:.3f} "
          f"| mean set {test['mean_set_size']:.2f} "
          f"| resolved {test['resolved_at_tau']:.3f}")
    print(f"bundle    {bundle_dir}")


def cmd_adapt(args):
    """Labeled data → calibrated bundle. The first command a user runs."""
    from src.decision.adapt import AdaptConfig, adapt
    from src.decision.datasets import read_labeled_file

    texts, labels = read_labeled_file(args.data, args.text_col, args.label_col)
    cfg = AdaptConfig(head_kind=args.head_kind, alpha=args.alpha,
                      min_cal=args.min_cal, seed=args.seed, policy=args.policy,
                      glial=args.glial,
                      head_kwargs=({"epochs": args.epochs} if args.epochs else {}))
    encoder = _encoder(args.backend, args.model, args.device, args.max_tokens,
                       args.ignore_memory)
    result = adapt(texts, labels, encoder, cfg=cfg, out_dir=args.out)

    # No ``default=`` here: ``adapt`` already writes this exact report with a
    # bare ``json.dumps``, so anything unserialisable should raise rather than be
    # quietly stringified — a number rendered as "0.93" is worse than a crash.
    if args.json:
        print(json.dumps(result.report, indent=2))
    else:
        _print_adapt_report(result.report, result.bundle_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nanocore", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    backend = _backend_args()
    model_args = _model_args()

    a = sub.add_parser("adapt", parents=[backend],
                       help="labeled data → calibrated bundle")
    a.add_argument("--data", required=True, help="CSV/TSV/JSONL of labeled examples")
    a.add_argument("--text-col", default="text", help="column holding the input text")
    a.add_argument("--label-col", default="label", help="column holding the class")
    a.add_argument("--out", required=True, help="output dir (writes bundle/ + report)")
    a.add_argument("--head-kind", default="linear", choices=["linear", "mlp"])
    a.add_argument("--alpha", type=float, default=0.10, help="miscoverage target")
    a.add_argument("--min-cal", type=int, default=200,
                   help="gate calibration floor; adapt fails below it")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--epochs", type=int, help="head training epochs (head default if unset)")
    a.add_argument("--policy", default="full", choices=["full", "escalate", "answer"])
    a.add_argument("--glial", action="store_true", help="attach the label-free slow state")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_adapt)

    d = sub.add_parser("decide", parents=[model_args],
                       help="one typed decision (per line with --inputs)")
    d.add_argument("text", nargs="?", default="-", help="input text (ignored with --inputs)")
    d.add_argument("--inputs", help="file with one input text per line")
    d.add_argument("--options", required=True, help="comma-separated option labels")
    d.add_argument("--qtype", default="choice", choices=["choice", "score", "noul"])
    d.add_argument("--scale", help="numeric scale for score questions")
    d.add_argument("--policy", default=None, choices=["full", "escalate", "answer"])
    d.add_argument("--bundle", help="calibrated bundle dir (enables the gate)")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_decide)

    r = sub.add_parser("refusals", parents=[model_args],
                       help="run the refusal battery (decision abstention)")
    r.add_argument("text", nargs="?", default="-", help="ignored; use --inputs")
    r.add_argument("--bundle", required=True, help="the calibrated bundle to test")
    r.add_argument("--options", required=True, help="the model's option labels")
    r.add_argument("--inputs", help="real texts for the invariants to run over")
    r.add_argument("--cases", help="JSON file of your own should-not-do rules")
    r.add_argument("--cfp", type=float, default=10.0,
                   help="cost of asserting a FALSE finding")
    r.add_argument("--cfn", type=float, default=20.0,
                   help="cost of discarding a REAL finding")
    r.add_argument("--cr", type=float, default=1.0, help="cost of routing to review")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_refusals)

    p = sub.add_parser("parity", parents=[backend],
                       help="compare two backends on the same inputs (Phase 8)")
    p.add_argument("text", nargs="?", default="-", help="input text (ignored with --inputs)")
    p.add_argument("--inputs", help="file with one input text per line")
    p.add_argument("--options", required=True, help="comma-separated option labels")
    p.add_argument("--bundle", help="the bundle both backends decide through")
    p.add_argument("--reference", default="st", choices=["st", "stub"],
                   help="the backend to compare against. NOTE: the default 'st' "
                        "loads PyTorch weights too — use 'stub' to exercise the "
                        "wiring without loading anything")
    p.add_argument("--min-cosine", type=float, default=0.99)
    p.add_argument("--min-action-agreement", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_parity)

    s = sub.add_parser("serve", parents=[model_args],
                       help="serve decisions over HTTP")
    s.add_argument("--bundle", help="calibrated bundle dir (enables the gate)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log", default=".nanocore-predictions.jsonl")
    s.add_argument("--policy", default=None, choices=["full", "escalate", "answer"])
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
