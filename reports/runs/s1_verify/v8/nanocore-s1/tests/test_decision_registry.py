"""Registry — versioned bundles, promotion, rollback, lineage."""
import json

import pytest

from src.decision.registry import Registry, data_fingerprint


def _fake_bundle(tmp_path, tag):
    d = tmp_path / f"bundle_{tag}"
    (d).mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"tag": tag}))
    return d


def test_register_and_promote(tmp_path):
    reg = Registry(tmp_path / "reg")
    v1 = reg.register("clinc", _fake_bundle(tmp_path, "a"), {"test": {"accuracy": 0.9}})
    v2 = reg.register("clinc", _fake_bundle(tmp_path, "b"), {"test": {"accuracy": 0.95}})
    assert (v1, v2) == ("v1", "v2")
    reg.promote("clinc", "v1")
    assert reg.current("clinc") == "v1"
    reg.promote("clinc", "v2")
    assert reg.current("clinc") == "v2"


def test_rollback_restores_previous(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.register("m", _fake_bundle(tmp_path, "a"), {})
    reg.register("m", _fake_bundle(tmp_path, "b"), {})
    reg.promote("m", "v1"); reg.promote("m", "v2")
    assert reg.rollback("m") == "v1"
    assert reg.current("m") == "v1"
    assert reg.rollback("m") == "v2"   # rollback is reversible


def test_lineage_records_parent_and_metrics(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.register("m", _fake_bundle(tmp_path, "a"), {"test": {"accuracy": 0.9}},
                 data_hash="abc123", config={"head_kind": "linear"})
    reg.promote("m", "v1")
    reg.register("m", _fake_bundle(tmp_path, "b"), {"test": {"accuracy": 0.95}})
    lin = reg.lineage("m", "v2")
    assert lin["parent"] == "v1"
    assert lin["metrics"]["accuracy"] == 0.95
    assert reg.lineage("m", "v1")["data_hash"] == "abc123"


def test_promote_missing_version_fails(tmp_path):
    reg = Registry(tmp_path / "reg")
    with pytest.raises(ValueError):
        reg.promote("m", "v9")


def test_data_fingerprint_stable(tmp_path):
    a = data_fingerprint(["x", "y"], ["p", "q"])
    b = data_fingerprint(["x", "y"], ["p", "q"])
    c = data_fingerprint(["x", "y"], ["p", "r"])
    assert a == b and a != c and len(a) == 16
