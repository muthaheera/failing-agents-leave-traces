"""
Langfuse Quickstart — 3 lines to behavioral drift scores in your dashboard
==========================================================================

Before running:
    pip install telemetry-eval[langfuse]
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...

What this does:
  1. Pulls your existing traced spans from Langfuse
  2. Computes behavioral fingerprints from OTel span data
  3. Detects drift between baseline and candidate releases using KS test
  4. Writes drift score back to Langfuse — appears in UI under "Scores"

No changes to your existing Langfuse setup required.
"""

from langfuse import Langfuse

from telemetry_eval import DriftDetector, extract_fingerprints
from telemetry_eval.exporters.langfuse import LangfuseExporter

# ── Your existing Langfuse client ────────────────────────────────────────────
langfuse = Langfuse()

# ── Step 1: Fetch spans from your existing Langfuse traces ───────────────────
# Replace these with your actual trace IDs or use your own fetch logic.
# Each "trace" is a list of OTel span dicts — standard fields only.

def fetch_spans_for_trace(trace_id: str) -> list[dict]:
    """
    Fetch OTel spans for a trace from Langfuse.
    Returns list of standard OTel span dicts.
    """
    trace = langfuse.fetch_trace(trace_id)
    spans = []
    for obs in trace.data.observations:
        # Map Langfuse observation fields to standard OTel span fields
        spans.append({
            "trace_id": obs.trace_id,
            "span_id": obs.id,
            "parent_span_id": obs.parent_observation_id,
            "name": obs.name or "unknown",
            "start_time": obs.start_time.timestamp() * 1e9 if obs.start_time else 0,
            "end_time": obs.end_time.timestamp() * 1e9 if obs.end_time else 0,
            "status": {"status_code": "ERROR" if obs.level == "ERROR" else "OK"},
            "attributes": {
                **(obs.metadata or {}),
                "gen_ai.usage.output_tokens": obs.usage.output if obs.usage else 0,
                "gen_ai.usage.input_tokens": obs.usage.input if obs.usage else 0,
            },
            "events": [],
        })
    return spans


# Example: replace with real trace IDs from your Langfuse project
BASELINE_TRACE_IDS = [
    "trace-baseline-001",
    "trace-baseline-002",
    # ... your baseline release trace IDs
]
CANDIDATE_TRACE_IDS = [
    "trace-candidate-001",
    "trace-candidate-002",
    # ... your candidate release trace IDs
]

# ── Step 2: Compute behavioral fingerprints (THE 3 LINES) ────────────────────

baseline_traces  = [fetch_spans_for_trace(tid) for tid in BASELINE_TRACE_IDS]
candidate_traces = [fetch_spans_for_trace(tid) for tid in CANDIDATE_TRACE_IDS]

baseline_fps  = extract_fingerprints(baseline_traces)   # line 1
candidate_fps = extract_fingerprints(candidate_traces)  # line 2
drift_result  = DriftDetector().detect(baseline_fps, candidate_fps)  # line 3

# ── Step 3: Write drift score back to Langfuse ───────────────────────────────

exporter = LangfuseExporter(langfuse)

for trace_id in CANDIDATE_TRACE_IDS:
    exporter.write_drift_score(trace_id, drift_result)

# ── Results ──────────────────────────────────────────────────────────────────

print(f"Drift label: {drift_result.label}")
print(f"Score:       {drift_result.score}  (1.0=no drift, 0.0=drift)")
print(f"KS stat:     {drift_result.composite_ks_statistic:.3f}")
print(f"p-value:     {drift_result.min_p_value:.4f}")
print(f"95% CI:      [{drift_result.ci_lower:.3f}, {drift_result.ci_upper:.3f}]")
print()
print("Per-signal breakdown:")
for sig in drift_result.signals:
    flag = "⚠ DRIFT" if sig.drifted else "✓ ok"
    print(f"  {sig.signal_name:35s}  {flag}  (delta: {sig.delta_pct:+.1f}%)")
print()
print(f"Drift scores written to {len(CANDIDATE_TRACE_IDS)} Langfuse traces.")
print("Open your Langfuse dashboard → Traces → any candidate trace → Scores tab.")
print()
print("Paper: https://www.amazon.science/publications/"
      "failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation")
