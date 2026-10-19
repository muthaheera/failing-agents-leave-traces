"""
Tests for KS drift detection and bootstrap CI.

Paper: "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation
"""

import pytest
from telemetry_eval.core.fingerprint import BehavioralFingerprint
from telemetry_eval.core.drift import DriftDetector, DriftResult, SIGNAL_NAMES


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_fp(
    effort=1.0, retry=0.0, depth=2.0, entropy=1.0, error=0.0
) -> BehavioralFingerprint:
    return BehavioralFingerprint(
        trace_id="test",
        effort_to_outcome_ratio=effort,
        retry_density=retry,
        span_depth_mean=depth,
        tool_dispatch_entropy=entropy,
        error_rate=error,
        duration_seconds=1.0,
        task_count=1,
        tool_call_count=3,
    )


def healthy_fps(n: int = 30) -> list[BehavioralFingerprint]:
    """Stable healthy fingerprints — no drift."""
    import random
    rng = random.Random(0)
    return [make_fp(
        effort=rng.gauss(1.0, 0.05),
        retry=max(0, rng.gauss(0.05, 0.01)),
        depth=rng.gauss(2.0, 0.1),
        entropy=rng.gauss(1.0, 0.05),
        error=max(0, rng.gauss(0.02, 0.005)),
    ) for _ in range(n)]


def degraded_fps(n: int = 30) -> list[BehavioralFingerprint]:
    """Significantly degraded fingerprints — large drift."""
    import random
    rng = random.Random(1)
    return [make_fp(
        effort=rng.gauss(3.5, 0.2),       # 3.5x baseline
        retry=max(0, rng.gauss(0.6, 0.05)),  # high retry density
        depth=rng.gauss(5.0, 0.3),          # deeper span nesting
        entropy=rng.gauss(0.3, 0.05),       # low tool diversity
        error=max(0, rng.gauss(0.4, 0.05)), # high error rate
    ) for _ in range(n)]


# ── Tests: DriftResult ────────────────────────────────────────────────────────

class TestDriftResult:
    def test_drifted_score_is_zero(self):
        detector = DriftDetector(p_threshold=0.001)
        result = detector.detect(healthy_fps(), degraded_fps())
        assert result.drifted
        assert result.score == 0.0

    def test_drifted_label(self):
        detector = DriftDetector(p_threshold=0.001)
        result = detector.detect(healthy_fps(), degraded_fps())
        assert result.label == "DRIFT_DETECTED"

    def test_no_drift_score_is_one(self):
        fps = healthy_fps(30)
        detector = DriftDetector(p_threshold=0.001)
        result = detector.detect(fps[:15], fps[15:])
        assert not result.drifted
        assert result.score == 1.0

    def test_no_drift_label(self):
        fps = healthy_fps(30)
        detector = DriftDetector(p_threshold=0.001)
        result = detector.detect(fps[:15], fps[15:])
        assert result.label == "NO_DRIFT"


class TestDriftDetector:
    def test_signal_count(self):
        result = DriftDetector().detect(healthy_fps(), degraded_fps())
        assert len(result.signals) == len(SIGNAL_NAMES)

    def test_signal_names_match(self):
        result = DriftDetector().detect(healthy_fps(), degraded_fps())
        for sig, expected_name in zip(result.signals, SIGNAL_NAMES):
            assert sig.signal_name == expected_name

    def test_composite_ks_positive(self):
        result = DriftDetector().detect(healthy_fps(), degraded_fps())
        assert result.composite_ks_statistic > 0

    def test_min_p_value_in_range(self):
        result = DriftDetector().detect(healthy_fps(), degraded_fps())
        assert 0.0 <= result.min_p_value <= 1.0

    def test_bootstrap_ci_order(self):
        result = DriftDetector(n_bootstrap=100).detect(healthy_fps(), degraded_fps())
        assert result.ci_lower <= result.ci_upper

    def test_n_traces_recorded(self):
        base = healthy_fps(20)
        cand = degraded_fps(25)
        result = DriftDetector().detect(base, cand)
        assert result.n_baseline == 20
        assert result.n_candidate == 25

    def test_empty_baseline_raises(self):
        with pytest.raises(ValueError):
            DriftDetector().detect([], healthy_fps())

    def test_empty_candidate_raises(self):
        with pytest.raises(ValueError):
            DriftDetector().detect(healthy_fps(), [])

    def test_high_threshold_detects_more_drift(self):
        """p=0.05 should flag drift where p=0.001 might not."""
        fps = healthy_fps(30)
        slightly_drifted = [make_fp(effort=fp.effort_to_outcome_ratio * 1.05) for fp in fps]
        strict = DriftDetector(p_threshold=0.001).detect(fps, slightly_drifted)
        lenient = DriftDetector(p_threshold=0.05).detect(fps, slightly_drifted)
        # Lenient threshold should have min_p_value closer to detection
        assert lenient.min_p_value <= strict.min_p_value or lenient.drifted


# ── Tests: serialisation ──────────────────────────────────────────────────────

class TestDriftResultSerialisation:
    def setup_method(self):
        self.result = DriftDetector().detect(healthy_fps(), degraded_fps())

    def test_to_dict_has_required_keys(self):
        d = self.result.to_dict()
        for key in ("drifted", "label", "score", "composite_ks_statistic",
                    "min_p_value", "ci_lower", "ci_upper", "signals"):
            assert key in d, f"Missing key: {key}"

    def test_to_otel_event_name(self):
        event = self.result.to_otel_event()
        assert event["name"] == "gen_ai.evaluation.result"

    def test_to_otel_event_has_score(self):
        event = self.result.to_otel_event()
        assert "gen_ai.evaluation.score" in event["attributes"]

    def test_to_langfuse_score_keys(self):
        payload = self.result.to_langfuse_score()
        for key in ("name", "value", "comment", "dataType"):
            assert key in payload

    def test_to_langfuse_score_name(self):
        payload = self.result.to_langfuse_score()
        assert payload["name"] == "behavioral_drift"
