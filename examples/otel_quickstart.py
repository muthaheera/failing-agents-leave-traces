"""
Pure OTel Quickstart — works with ANY OTel-compatible backend
=============================================================
Langfuse · Arize Phoenix · LangSmith · Honeycomb · Grafana · Datadog · Jaeger

Before running:
    pip install telemetry-eval opentelemetry-sdk opentelemetry-exporter-otlp

What this does:
  Instruments a simple two-agent pipeline, extracts behavioral fingerprints
  from the OTel spans, detects drift, and emits the result as a standard
  gen_ai.evaluation.result span event — readable by any OTLP backend.
"""

import time
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from telemetry_eval import (
    BehavioralFingerprint,
    DriftDetector,
    ReleaseGate,
    ROPOLLJudge,
    extract_fingerprint,
    extract_fingerprints,
)
from telemetry_eval.exporters.otel import drift_to_span_event, fingerprint_to_span_attributes

# ── OTel setup — swap ConsoleSpanExporter for your OTLP exporter ─────────────
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("telemetry-eval-example")


# ── Simulate agent spans ──────────────────────────────────────────────────────

def make_fake_span(name: str, parent_id: str | None = None, is_tool: bool = False,
                   is_error: bool = False, duration_ms: float = 100.0) -> dict:
    """Build a minimal OTel span dict (mimics what any instrumentation library produces)."""
    now = time.time_ns()
    return {
        "trace_id": "trace-" + uuid.uuid4().hex[:16],
        "span_id": "span-" + uuid.uuid4().hex[:8],
        "parent_span_id": parent_id,
        "name": name,
        "start_time": now,
        "end_time": now + int(duration_ms * 1e6),
        "status": {"status_code": "ERROR" if is_error else "OK"},
        "attributes": {
            "execute_tool": is_tool,
            "gen_ai.usage.output_tokens": 120 if not is_tool else 0,
        },
        "events": [],
    }


def make_baseline_trace() -> list[dict]:
    """A healthy 3-span trace: plan → tool → respond."""
    root = make_fake_span("invoke_agent", duration_ms=300)
    tool = make_fake_span("execute_tool", parent_id=root["span_id"], is_tool=True, duration_ms=80)
    resp = make_fake_span("invoke_agent.respond", parent_id=root["span_id"], duration_ms=150)
    return [root, tool, resp]


def make_degraded_trace() -> list[dict]:
    """A degraded trace: excessive retries, errors, deeper span nesting."""
    root = make_fake_span("invoke_agent", duration_ms=900)
    t1   = make_fake_span("execute_tool", parent_id=root["span_id"], is_tool=True, duration_ms=200)
    t1r  = make_fake_span("execute_tool.retry", parent_id=t1["span_id"], is_tool=True, duration_ms=200)
    err  = make_fake_span("execute_tool.retry.2", parent_id=t1r["span_id"], is_tool=True, is_error=True)
    t2   = make_fake_span("execute_tool", parent_id=root["span_id"], is_tool=True, duration_ms=300)
    resp = make_fake_span("invoke_agent.respond", parent_id=root["span_id"], is_error=True, duration_ms=400)
    return [root, t1, t1r, err, t2, resp]


# ── Build baseline and candidate trace sets ───────────────────────────────────

baseline_traces  = [make_baseline_trace() for _ in range(30)]
candidate_traces = [make_degraded_trace() for _ in range(30)]

# ── THE 3 LINES ───────────────────────────────────────────────────────────────

baseline_fps  = extract_fingerprints(baseline_traces)   # line 1
candidate_fps = extract_fingerprints(candidate_traces)  # line 2
drift_result  = DriftDetector().detect(baseline_fps, candidate_fps)  # line 3

# ── Emit result as OTel span event ────────────────────────────────────────────

with tracer.start_as_current_span("telemetry_eval.drift_check") as span:
    # Attach fingerprint signals as span attributes
    fp_sample = candidate_fps[0]
    span.set_attributes(fingerprint_to_span_attributes(fp_sample))

    # Emit drift result as gen_ai.evaluation.result event
    event = drift_to_span_event(drift_result)
    span.add_event(event["name"], event["attributes"])

# ── ROPOLL judge example ──────────────────────────────────────────────────────

def mock_judge(input_text: str, output_text: str) -> float:
    """Replace with your real OpenAI / Anthropic judge call."""
    import random
    return random.gauss(0.75, 0.1)

ropoll = ROPOLLJudge(judge_fn=mock_judge, n_samples=5)
judge_result = ropoll.evaluate(
    input_text="Summarise this document in 3 bullet points.",
    output_text="• Key finding 1\n• Key finding 2\n• Key finding 3",
)

# ── Print summary ─────────────────────────────────────────────────────────────

print("=" * 60)
print(f"Behavioral Drift: {drift_result.label}")
print(f"Score:            {drift_result.score}")
print(f"KS stat:          {drift_result.composite_ks_statistic:.3f}")
print(f"p-value:          {drift_result.min_p_value:.4f}")
print(f"95% CI:           [{drift_result.ci_lower:.3f}, {drift_result.ci_upper:.3f}]")
print()
print("Per-signal breakdown:")
for sig in drift_result.signals:
    flag = "⚠ DRIFT" if sig.drifted else "✓ ok"
    print(f"  {sig.signal_name:35s}  {flag}  (Δ {sig.delta_pct:+.1f}%)")
print()
print(f"ROPOLL judge score: {judge_result.robust_score:.3f}  "
      f"(naive mean: {judge_result.naive_mean:.3f})")
if judge_result.judge_drift_detected:
    print(f"  ⚠ Judge drift detected — naive averaging would have been wrong by {judge_result.judge_drift_delta:.3f}")
print()
print("OTel events emitted. Send spans to any OTLP receiver:")
print("  Langfuse, Arize Phoenix, LangSmith, Honeycomb, Grafana, Datadog…")
print()
print("Paper: https://www.amazon.science/publications/"
      "failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation")
