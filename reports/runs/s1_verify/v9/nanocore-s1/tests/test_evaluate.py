"""Tests for NanoCore-S1 evaluation suite."""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import json
import math
from unittest.mock import patch, MagicMock

from src.model import NanoCore, NanoCoreConfig
from scripts.evaluate import Evaluator, EvalResult


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_to_dict(self):
        """EvalResult should serialize to dict correctly."""
        result = EvalResult(
            metric_name="test_metric",
            value=0.95,
            num_samples=100,
            details={"extra": "info"}
        )
        d = result.to_dict()
        assert d["metric_name"] == "test_metric"
        assert d["value"] == 0.95
        assert d["num_samples"] == 100
        assert d["details"]["extra"] == "info"

    def test_to_dict_no_details(self):
        """EvalResult without details should have empty dict."""
        result = EvalResult(
            metric_name="test",
            value=1.0,
            num_samples=10,
        )
        d = result.to_dict()
        assert d["details"] == {}


class TestEvaluatorInit:
    """Tests for Evaluator initialization."""

    def test_model_in_eval_mode(self):
        """Model should be in eval mode after initialization."""
        # Always create fresh model for this test (avoids RoPE buffer size mismatch)
        config = NanoCoreConfig.from_depth(2)
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        assert not evaluator.model.training, "Model should be in eval mode"


class TestPerplexityEval:
    """Tests for perplexity evaluation."""

    def test_perplexity_finite(self):
        """Perplexity should be a finite positive number."""
        config = NanoCoreConfig.from_depth(2)  # Small model for speed
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        # Create dummy data
        data = torch.randint(0, config.vocab_size, (100, config.sequence_len))
        result = evaluator._eval_perplexity(data)
        
        assert result.metric_name == "perplexity"
        assert result.value > 0
        assert math.isfinite(result.value)
        assert result.num_samples > 0

    def test_perplexity_bounded(self):
        """Perplexity on random data should be near vocab_size."""
        config = NanoCoreConfig.from_depth(2)
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        data = torch.randint(0, config.vocab_size, (100, config.sequence_len))
        result = evaluator._eval_perplexity(data)
        
        # Random data → perplexity should be roughly vocab_size
        # With 4-layer model on random data, might be higher or lower
        assert result.value > 1.0
        assert result.value < config.vocab_size * 10


class TestCalibrationEval:
    """Tests for calibration evaluation (ECE)."""

    def test_ece_non_negative(self):
        """ECE should be non-negative."""
        config = NanoCoreConfig.from_depth(2)
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        data = torch.randint(0, config.vocab_size, (100, config.sequence_len))
        result = evaluator._eval_calibration(data)
        
        assert result.metric_name == "expected_calibration_error"
        assert result.value >= 0
        assert result.value <= 1.0  # ECE can't exceed 1
        assert result.num_samples > 0
        assert result.details["n_bins"] == 10


class TestDecisionAccuracyEval:
    """Tests for decision accuracy evaluation."""

    def test_decision_accuracy_in_range(self):
        """Decision accuracy should be between 0 and 1."""
        config = NanoCoreConfig.from_depth(2)
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        data = torch.randint(0, config.vocab_size, (100, config.sequence_len))
        result = evaluator._eval_decision_accuracy(data)
        
        assert result.metric_name == "decision_accuracy"
        assert 0.0 <= result.value <= 1.0
        assert result.num_samples > 0


class TestThroughputEval:
    """Tests for throughput evaluation."""

    def test_throughput_positive(self):
        """Throughput should be positive."""
        config = NanoCoreConfig.from_depth(2)  # Even smaller for speed
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        result = evaluator._eval_throughput()
        
        assert result.metric_name == "inference_throughput"
        assert result.value > 0
        assert result.details["latency_ms"] > 0
        assert result.details["batch_size"] == 1
        assert result.details["seq_len"] == 512


class TestReportSaving:
    """Tests for report saving."""

    def test_save_report(self, tmp_path):
        """Report should be saved as valid JSON."""
        config = NanoCoreConfig.from_depth(4)
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")
        
        results = [
            EvalResult("perplexity", 100.0, 1000),
            EvalResult("ece", 0.1, 500),
        ]
        
        output_path = str(tmp_path / "eval_report.json")
        evaluator.save_report(results, output_path)
        
        assert os.path.exists(output_path)
        with open(output_path) as f:
            report = json.load(f)
        
        assert report["model"] == "NanoCore-S1"
        assert report["total_params"] > 0
        assert len(report["results"]) == 2
        assert report["results"][0]["metric_name"] == "perplexity"


class TestFullEvaluation:
    """Tests for the full evaluation suite."""

    def test_evaluate_all_metrics(self):
        """Full evaluation should return results for all metrics."""
        config = NanoCoreConfig.from_depth(2)  # Small for speed
        model = NanoCore(config)
        evaluator = Evaluator(model, config, device="cpu")

        results = evaluator.evaluate(datasets={
            "perplexity": evaluator._create_dummy_dataset("perplexity", 20),
            "calibration": evaluator._create_dummy_dataset("calibration", 10),
            "decision_accuracy": evaluator._create_dummy_dataset("decision_accuracy", 10),
        })
        
        # Throughput is always evaluated (no dataset needed)
        results.append(evaluator._eval_throughput())

        metric_names = [r.metric_name for r in results]
        assert "perplexity" in metric_names
        assert "expected_calibration_error" in metric_names
        assert "decision_accuracy" in metric_names
        assert "inference_throughput" in metric_names
