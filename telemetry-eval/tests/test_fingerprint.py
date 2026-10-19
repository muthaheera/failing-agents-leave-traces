"""
Tests for behavioral fingerprint extraction.

Paper: "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation
"""

import pytest
from telemetry_eval.core.fingerprint import (
    BehavioralFingerprint,
    extract_fingerprint,
    extract_fingerprints,
    _is_tool_span,
    _is_retry,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

_SPAN_COUNTER = [0]

def make_span(
    name: str = "invoke_agent",
    parent_id: str | None = None,
    duration_ms: float = 100.0,
    is_error: bool = False,
    output_tokens: int = 50,
    is_retry: bool = False,
) -> dict:
    """Minimal OTel span dict for testing."""
    import time
    _SPAN_COUNTER[0] += 1
    now = int(time.time() * 1e9)
    attrs: dict = {"gen_ai.usage.output_tokens": output_tokens}
    if is_retry:
        attrs["retry_count"] = 1  # _is_retry checks attributes, not span name
    status_str = "ERROR" if is_error else "OK"
    return {
        "trace_id": "trace-test",
        "span_id": f"span-{_SPAN_COUNTER[0]}",
        "parent_span_id": parent_id,
        "name": name,
        "start_time": now,
        "end_time": now + int(duration_ms * 1e6),
        "status": status_str,          # fingerprint.py checks status directly as string
        "attributes": attrs,
        "events": [],
    }


def healthy_trace() -> list[dict]:
    """Simple healthy trace: root → tool → respond."""
    root = make_span("invoke_agent", duration_ms=300, output_tokens=100)
    tool = make_span("execute_tool", parent_id=root["span_id"], duration_ms=50, output_tokens=0)
    resp = make_span("invoke_agent.respond", parent_id=root["span_id"], duration_ms=150, output_tokens=80)
    return [root, tool, resp]


def degraded_trace() -> list[dict]:
    """Degraded trace: many retries + errors."""
    root  = make_span("invoke_agent", duration_ms=1200, output_tokens=20)
    t1    = make_span("execute_tool", parent_id=root["span_id"], duration_ms=200,
                      is_retry=True)  # retry_count attr triggers _is_retry
    t1r   = make_span("execute_tool", parent_id=t1["span_id"], duration_ms=200,
                      is_retry=True)
    err   = make_span("execute_tool", parent_id=t1r["span_id"],
                      is_error=True, is_retry=True, output_tokens=0)
    t2    = make_span("execute_tool", parent_id=root["span_id"], duration_ms=300)
    resp  = make_span("invoke_agent.respond", parent_id=root["span_id"],
                      is_error=True, output_tokens=10)
    return [root, t1, t1r, err, t2, resp]


# ── Tests: extraction ─────────────────────────────────────────────────────────

class TestExtractFingerprint:
    def test_returns_behavioral_fingerprint(self):
        fp = extract_fingerprint(healthy_trace())
        assert isinstance(fp, BehavioralFingerprint)

    def test_healthy_trace_low_error_rate(self):
        fp = extract_fingerprint(healthy_trace())
        assert fp.error_rate == 0.0

    def test_degraded_trace_has_errors(self):
        fp = extract_fingerprint(degraded_trace())
        assert fp.error_rate > 0.0

    def test_degraded_trace_has_retries(self):
        fp = extract_fingerprint(degraded_trace())
        assert fp.retry_density > 0.0

    def test_healthy_trace_zero_retry_density(self):
        fp = extract_fingerprint(healthy_trace())
        assert fp.retry_density == 0.0

    def test_tool_count_correct(self):
        fp = extract_fingerprint(healthy_trace())
        # 1 tool span (execute_tool)
        assert fp.tool_call_count >= 1

    def test_span_depth_positive(self):
        fp = extract_fingerprint(healthy_trace())
        assert fp.span_depth_mean > 0

    def test_empty_trace_returns_zero_fingerprint(self):
        fp = extract_fingerprint([])
        assert fp.signal_vector() == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_signal_vector_has_five_elements(self):
        fp = extract_fingerprint(healthy_trace())
        vec = fp.signal_vector()
        assert len(vec) == 5

    def test_all_signals_non_negative(self):
        fp = extract_fingerprint(degraded_trace())
        for v in fp.signal_vector():
            assert v >= 0.0, f"Signal value {v} is negative"


class TestExtractFingerprints:
    def test_batch_returns_list(self):
        traces = [healthy_trace(), degraded_trace(), healthy_trace()]
        fps = extract_fingerprints(traces)
        assert len(fps) == 3

    def test_batch_matches_individual(self):
        t = healthy_trace()
        single = extract_fingerprint(t)
        batch = extract_fingerprints([t])
        assert batch[0].signal_vector() == single.signal_vector()


# ── Tests: span classification helpers ───────────────────────────────────────

class TestSpanClassification:
    def test_tool_span_by_name(self):
        span = make_span("execute_tool")
        assert _is_tool_span(span)

    def test_non_tool_span(self):
        span = make_span("invoke_agent")
        assert not _is_tool_span(span)

    def test_retry_span_by_attribute(self):
        span = make_span("execute_tool", is_retry=True)
        assert _is_retry(span)

    def test_non_retry_span(self):
        span = make_span("invoke_agent")
        assert not _is_retry(span)
