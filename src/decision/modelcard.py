"""The model card — the artifact a recipient of this model expects to receive.

Mitchell et al. (2019) proposed nine sections; Hugging Face adopted them as the
de facto delivery convention, with a structured ``model-index`` block in the
YAML front matter that the Hub parses into an evaluation widget. Neither has a
conformance test — a document with two of nine sections filled in is still
"a model card" — so this generator writes all nine, and refuses to emit one with
empty required sections rather than shipping a plausible-looking shell.

Two things are deliberately *not* softened:

- **Out-of-scope uses are stated as limits, not caveats.** A decision layer that
  has never been tested on a modality or a language must say so, because the
  failure mode of a model card is a reader assuming coverage that was never
  measured.
- **Unmeasured criteria appear as "not measured".** The gate's ``deferred``
  status is carried through verbatim; a card that renders a deferral as a pass
  would be worse than no card.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

#: Sections Mitchell et al. propose. A card missing any of these is not emitted.
REQUIRED_SECTIONS = ("model_details", "intended_use", "factors", "metrics",
                     "evaluation_data", "training_data", "quantitative_analyses",
                     "ethical_considerations", "caveats")


def _bullet(items) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "- (none recorded)"


def _table(rows: List[List[str]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def model_index(name: str, metrics: List[Dict]) -> Dict:
    """The structured block the Hub parses. One entry per (task, dataset, metric)."""
    return {"name": name,
            "results": [{"task": {"type": r.get("task", "text-classification"),
                                  "name": r.get("task_name")},
                         "dataset": {"name": r["dataset"], "type": r.get("dataset_type",
                                                                         r["dataset"])},
                         "metrics": [{"type": r["metric"], "value": r["value"]}]}
                        for r in metrics]}


def _metrics_from_adapt(adapt_report: Optional[Dict], dataset: str) -> List[Dict]:
    """Hub-shaped eval rows from an adapt report's held-out block."""
    test = (adapt_report or {}).get("test", {})
    return [{"dataset": dataset or "user-supplied", "metric": metric, "value": test[key]}
            for key, metric in (("accuracy", "accuracy"), ("brier", "brier"),
                                ("conformal_coverage", "coverage"),
                                ("mean_set_size", "mean set size"))
            if key in test]


def _analyses_from_qualification(qualification: Dict) -> Dict:
    """The quantitative-analysis section, read off the gate's own statuses.

    Kept separate from the document literals because this is the part with
    logic: a status the gate did not measure must never be summarised as one it
    did, and that distinction is exactly one `if` away from being lost.
    """
    criteria = qualification.get("criteria", [])
    by_status: Dict[str, List[Dict]] = {}
    for c in criteria:
        by_status.setdefault(c["status"], []).append(c)
    return {
        "verdict": qualification.get("verdict", "unknown"),
        "passed": len(by_status.get("pass", [])),
        "failed": [c["id"] for c in by_status.get("fail", [])],
        "open_capability_gaps": [c["id"] for c in criteria
                                 if c["severity"] == "must_fix" and c["status"] != "pass"],
        "not_measured": [c["id"] for c in by_status.get("deferred", [])],
        "criteria": criteria,
    }


def build_card(qualification: Dict, adapt_report: Optional[Dict] = None,
               *, model_name: str = "nanocore-s1", dataset: str = "",
               license: str = "apache-2.0",
               encoder_id: str = "google/embeddinggemma-2") -> Dict:
    """Assemble the card's content as data, before any rendering.

    Returning a dict keeps the sections testable — a rendered string can only be
    checked by eye, which is how a card ends up with an empty "Limitations".
    """
    prov = qualification.get("provenance", {}) or {}
    metrics = _metrics_from_adapt(adapt_report, dataset)

    card = {
        "model_details": {
            "name": model_name,
            "version": prov.get("git_commit", "unknown"),
            "type": "frozen multimodal encoder + trained head + conformal gate",
            "encoder": encoder_id,
            "artifact_digest": prov.get("bundle_sha256") or "NOT RECORDED",
            "license": license,
            "citation": "EmbeddingGemma 2 (Google); conformal prediction (Vovk et al.)",
        },
        "intended_use": {
            "primary": "typed single-label decisions over a caller-supplied option set "
                       "(answer / clarify / escalate / abstain) with a coverage guarantee",
            "users": "an application embedding a decision step; a human reviewing escalations",
            "out_of_scope": [
                "generation of any kind — this model emits labels, not text",
                "open-ended option sets it was not adapted to (it will escalate, by design)",
                "audio and video states: the encoder supports them, this build has never run them",
                "languages outside the measured set — see the language gap below",
                "any use where an escalation has nowhere to go",
            ],
        },
        "factors": {
            "measured": ["language (en + de/es/fr/ru/zh-CN/ja)",
                         "option-set size (2-150 measured)",
                         "modality (text; vision at decision level on one task)"],
            "not_measured": ["demographic or user-attribute slices",
                             "domain shift beyond the measured corpora",
                             "adversarial input"],
        },
        "metrics": {
            "why": "proper scoring rules gate; ECE is reported but never gated, because "
                   "sharpening minimises ECE without improving the distribution",
            "gated": ["log score", "Brier", "conformal coverage", "mean set size",
                      "refusal behaviour"],
            "reported_not_gated": ["ECE (15 bins)", "AURC", "selective accuracy"],
        },
        "evaluation_data": {
            "datasets": dataset or "see qualification report",
            "splits": "fit / calibrate / evaluate, disjoint; thresholds chosen on the "
                      "first two and applied unchanged to the third",
            "seed": (adapt_report or {}).get("splits", {}).get("seed", "recorded in the report"),
        },
        "training_data": {
            "note": "only the head and the gate are fitted; the encoder is frozen",
            "source": (adapt_report or {}).get("n_total", "user-supplied labeled data"),
            "classes": (adapt_report or {}).get("n_classes", "recorded in the report"),
            "headroom": (adapt_report or {}).get("headroom", {}).get("zeroshot_test_acc"),
        },
        "quantitative_analyses": _analyses_from_qualification(qualification),
        "ethical_considerations": {
            "privacy": "inputs never leave the device; the decision path opens no "
                       "socket (tested), and the GGUF backend is local-only",
            "failure_mode": "the tolerable failure is an abstention, not a confident "
                            "wrong answer — refusals are gated on both sides, because a "
                            "model that refuses everything is also broken (XSTest)",
            "bias": "not measured beyond language; state this rather than imply coverage",
        },
        "caveats": {
            "known_limitations": [
                "two-field / relational composition is weak (mnli below the gate floor)",
                "mean-pooling a multi-item state does not beat the best single item",
                "conformal sets are uninformative on tasks where the scorer has no signal",
                "no offline-to-online correlation: no deployment has been measured",
            ],
            "independence": "validation is currently performed by the same process that "
                            "builds the model; SR 11-7 requires independence and this is "
                            "the largest outstanding gap",
        },
    }

    missing = [s for s in REQUIRED_SECTIONS if not card.get(s)]
    if missing:
        raise ValueError(f"card is missing required sections: {missing}")
    return {"card": card, "metrics": metrics,
            "model_index": model_index(model_name, metrics) if metrics else None}


def render_markdown(built: Dict) -> str:
    """Render the card as Markdown with YAML front matter, Hub-readable."""
    card, metrics = built["card"], built["metrics"]
    mi = built.get("model_index")
    front = ["---", f"license: {card['model_details']['license']}"]
    if mi:
        front.append("model-index:")
        front.append(json.dumps([mi], indent=2).replace("\n", "\n  ").lstrip())
    front.append("---")

    lines = ["\n".join(front), "", f"# {card['model_details']['name']}", ""]
    for section in REQUIRED_SECTIONS:
        body = card[section]
        lines += [f"## {section.replace('_', ' ').title()}", ""]
        for key, value in body.items():
            lines.append(f"**{key.replace('_', ' ')}**")
            if isinstance(value, list):
                lines.append(_bullet(value))
            elif isinstance(value, dict):
                lines.append("```json\n" + json.dumps(value, indent=2, default=str) + "\n```")
            else:
                lines.append(str(value))
            lines.append("")
    if metrics:
        lines += ["## Evaluation results", "",
                  _table([[m["dataset"], m["metric"], m["value"]] for m in metrics],
                         ["dataset", "metric", "value"]), ""]
    return "\n".join(lines)
