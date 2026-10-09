"""`nanocore adapt` — the first command a user runs, end to end and weightless.

Definition-of-done #1 is "a person can install it, point it at their labeled
data, and get a calibrated bundle". Before this command existed, `adapt` was
reachable only from Python, so a user could consume a bundle but not make one.

Everything here runs on `--backend stub` (no weights) so it stays local and fast.
"""
import json

import pytest

from src.decision.cli import main
from src.decision.datasets import read_labeled_file

CLASSES = ["cancel", "book", "balance"]
PER_CLASS = 60


def _write_csv(path):
    rows = ["text,label"]
    for c in CLASSES:
        for j in range(PER_CLASS):
            rows.append(f"{c} request number {j},{c}")
    path.write_text("\n".join(rows) + "\n")
    return path


def _write_jsonl(path):
    lines = [json.dumps({"utterance": f"{c} request {j}", "intent": c})
             for c in CLASSES for j in range(PER_CLASS)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_adapt_writes_a_bundle_and_a_report(tmp_path, capsys):
    data = _write_csv(tmp_path / "tickets.csv")
    out = tmp_path / "run"
    main(["adapt", "--data", str(data), "--out", str(out), "--backend", "stub",
          "--min-cal", "20", "--epochs", "3"])
    printed = capsys.readouterr().out
    assert "classes" in printed and "accuracy" in printed

    assert (out / "bundle" / "manifest.json").exists()
    report = json.loads((out / "adapt_report.json").read_text())
    assert report["n_total"] == len(CLASSES) * PER_CLASS
    assert report["n_classes"] == len(CLASSES)
    assert report["test"]["n"] > 0


def test_the_bundle_adapt_produced_can_be_decided_with(tmp_path, capsys):
    """The contract that matters: the artifact one command writes is the one the
    next command reads. A bundle that cannot be served is not a bundle."""
    data = _write_csv(tmp_path / "tickets.csv")
    out = tmp_path / "run"
    main(["adapt", "--data", str(data), "--out", str(out), "--backend", "stub",
          "--min-cal", "20", "--epochs", "3"])
    capsys.readouterr()

    main(["decide", "cancel request number 7", "--options", ",".join(CLASSES),
          "--backend", "stub", "--bundle", str(out / "bundle"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] in ("answer", "clarify", "escalate", "abstain")
    assert set(payload["probabilities"]) == set(CLASSES)


def test_jsonl_and_custom_columns(tmp_path, capsys):
    data = _write_jsonl(tmp_path / "tickets.jsonl")
    out = tmp_path / "run"
    main(["adapt", "--data", str(data), "--text-col", "utterance",
          "--label-col", "intent", "--out", str(out), "--backend", "stub",
          "--min-cal", "20", "--epochs", "3", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["n_classes"] == len(CLASSES)


def test_adapt_fails_closed_below_the_calibration_floor(tmp_path):
    data = _write_csv(tmp_path / "tickets.csv")
    with pytest.raises(ValueError, match="calibration split too small"):
        main(["adapt", "--data", str(data), "--out", str(tmp_path / "run"),
              "--backend", "stub", "--min-cal", "100000", "--epochs", "2"])


def test_missing_column_is_named_not_guessed(tmp_path, capsys):
    """Guessing which column is the label is how a feature becomes a target."""
    data = tmp_path / "tickets.csv"
    data.write_text("text,verdict\na,b\n")
    with pytest.raises(ValueError, match="not in"):
        main(["adapt", "--data", str(data), "--out", str(tmp_path / "run"),
              "--backend", "stub"])


def test_read_labeled_file_rejects_an_empty_file(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        read_labeled_file(empty)


def test_read_labeled_file_missing_path():
    with pytest.raises(FileNotFoundError, match="no such data file"):
        read_labeled_file("/nonexistent/nope.csv")


def test_labels_are_not_coerced(tmp_path):
    """A numeric-looking label column is the user's own class names, not ids."""
    data = tmp_path / "n.csv"
    data.write_text("text,label\nhello,0\nworld,1\n")
    texts, labels = read_labeled_file(data)
    assert labels == ["0", "1"] and texts == ["hello", "world"]
