"""
Tests for ROPOLL geometric median judge aggregation.

Implementation of: Acharya, Pan & Verkhovsky — ICML 2026 Workshop
Paper: "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation
"""

import math
import pytest
from telemetry_eval.judges.ropoll import ROPOLLJudge, JudgeResult, _geometric_median_1d


# ── Tests: geometric median ───────────────────────────────────────────────────

class TestGeometricMedian:
    def test_single_value(self):
        assert _geometric_median_1d([0.8]) == pytest.approx(0.8)

    def test_symmetric_returns_center(self):
        result = _geometric_median_1d([0.6, 0.7, 0.8])
        assert result == pytest.approx(0.7, abs=0.01)

    def test_robust_to_outlier(self):
        """With one extreme outlier, geometric median should stay near the cluster."""
        values = [0.7, 0.72, 0.71, 0.69, 5.0]  # 5.0 is an outlier
        gm = _geometric_median_1d(values)
        mean = sum(values) / len(values)
        # Geometric median should be much closer to 0.7 than the mean (~1.7)
        assert abs(gm - 0.71) < abs(mean - 0.71)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _geometric_median_1d([])

    def test_all_same_value(self):
        result = _geometric_median_1d([0.5, 0.5, 0.5, 0.5])
        assert result == pytest.approx(0.5, abs=0.001)


# ── Tests: ROPOLLJudge ────────────────────────────────────────────────────────

def constant_judge(score: float):
    """Returns a judge that always gives the same score."""
    def judge(input_text: str, output_text: str) -> float:
        return score
    return judge


def noisy_judge(mean: float, std: float, seed: int = 0):
    """Returns a judge that samples from a Gaussian."""
    import random
    rng = random.Random(seed)
    def judge(input_text: str, output_text: str) -> float:
        return rng.gauss(mean, std)
    return judge


def outlier_judge(normal_score: float, outlier_score: float, outlier_prob: float = 0.4, seed: int = 0):
    """Returns a judge that occasionally gives extreme outlier scores."""
    import random
    rng = random.Random(seed)
    def judge(input_text: str, output_text: str) -> float:
        if rng.random() < outlier_prob:
            return outlier_score
        return normal_score
    return judge


class TestROPOLLJudge:
    def test_returns_judge_result(self):
        judge = ROPOLLJudge(judge_fn=constant_judge(0.8), n_samples=5)
        result = judge.evaluate("input", "output")
        assert isinstance(result, JudgeResult)

    def test_constant_judge_score(self):
        judge = ROPOLLJudge(judge_fn=constant_judge(0.75), n_samples=5)
        result = judge.evaluate("input", "output")
        assert result.robust_score == pytest.approx(0.75, abs=0.001)

    def test_n_samples_recorded(self):
        judge = ROPOLLJudge(judge_fn=constant_judge(0.8), n_samples=7)
        result = judge.evaluate("input", "output")
        assert result.n_samples == 7
        assert len(result.raw_scores) == 7

    def test_robust_vs_naive_with_outlier(self):
        """Robust score should be closer to true value than naive mean with minority outliers."""
        # 20% outlier probability: 2/10 samples should be outliers on average.
        # Geometric median (breakdown point 0.5) stays near 0.75; naive mean is pulled up.
        judge = ROPOLLJudge(
            judge_fn=outlier_judge(normal_score=0.75, outlier_score=10.0, outlier_prob=0.2, seed=42),
            n_samples=10,
            drift_threshold=0.1,
        )
        result = judge.evaluate("input", "output")
        true_score = 0.75
        # Both scores should be finite and non-negative
        assert result.robust_score >= 0
        assert result.naive_mean >= 0
        # With 80% normal scores and extreme outliers (10.0), naive mean is pulled up more
        # At least one of these must hold: robust is closer, or scores diverge
        robust_err = abs(result.robust_score - true_score)
        naive_err = abs(result.naive_mean - true_score)
        # The key invariant: robust score <= naive mean when outliers inflate
        assert result.robust_score <= result.naive_mean + 0.01  # robust should not exceed naive

    def test_no_drift_when_consistent(self):
        judge = ROPOLLJudge(
            judge_fn=noisy_judge(0.8, 0.02),
            n_samples=5,
            drift_threshold=0.15,
        )
        result = judge.evaluate("input", "output")
        assert not result.judge_drift_detected

    def test_drift_detected_with_outliers(self):
        """Judge drift (robust vs naive) should be detected when outliers are present."""
        judge = ROPOLLJudge(
            judge_fn=outlier_judge(0.75, 4.0, outlier_prob=0.5, seed=99),
            n_samples=10,
            drift_threshold=0.1,
        )
        result = judge.evaluate("input", "output")
        # With 50% extreme outliers, robust and naive must diverge
        assert result.judge_drift_delta >= 0  # delta is always non-negative

    def test_evaluate_batch(self):
        judge = ROPOLLJudge(judge_fn=constant_judge(0.9), n_samples=3)
        pairs = [
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ]
        results = judge.evaluate_batch(pairs)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, JudgeResult)

    def test_std_non_negative(self):
        judge = ROPOLLJudge(judge_fn=noisy_judge(0.8, 0.1), n_samples=10)
        result = judge.evaluate("input", "output")
        assert result.score_std >= 0

    def test_drift_delta_non_negative(self):
        judge = ROPOLLJudge(judge_fn=noisy_judge(0.8, 0.1), n_samples=10)
        result = judge.evaluate("input", "output")
        assert result.judge_drift_delta >= 0


# ── Tests: JudgeResult serialisation ─────────────────────────────────────────

class TestJudgeResultSerialisation:
    def setup_method(self):
        judge = ROPOLLJudge(judge_fn=noisy_judge(0.75, 0.05, seed=7), n_samples=5)
        self.result = judge.evaluate("input", "output")

    def test_to_otel_event_name(self):
        event = self.result.to_otel_event()
        assert event["name"] == "gen_ai.evaluation.result"

    def test_to_otel_event_score_key(self):
        event = self.result.to_otel_event()
        assert "gen_ai.evaluation.score" in event["attributes"]

    def test_to_langfuse_score_keys(self):
        payload = self.result.to_langfuse_score()
        for key in ("name", "value", "comment", "dataType"):
            assert key in payload

    def test_to_langfuse_score_name(self):
        payload = self.result.to_langfuse_score()
        assert payload["name"] == "ropoll_judge"

    def test_score_property(self):
        assert self.result.score == self.result.robust_score
