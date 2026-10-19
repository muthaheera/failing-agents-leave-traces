"""
OTel Exporter — works with ANY OTel-compatible backend
=======================================================
Emits behavioral drift results as standard OpenTelemetry span events
using the gen_ai.evaluation.result semantic convention.

Compatible with: Langfuse, Arize Phoenix, LangSmith, Honeycomb,
                 Grafana Tempo, Jaeger, Datadog, or any OTLP receiver.

No vendor SDK required — pure OTel.
"""

from __future__ import annotations

from typing import Any

from telemetry_eval.core.drift import DriftResult
from telemetry_eval.judges.ropoll import JudgeResult


def drift_to_span_event(result: DriftResult) -> dict[str, Any]:
    """
    Convert a DriftResult to an OTel span event dict.

    Usage
    -----
    >>> from opentelemetry import trace
    >>> span = trace.get_current_span()
    >>> span.add_event(**drift_to_span_event(drift_result))
    """
    return result.to_otel_event()


def judge_to_span_event(result: JudgeResult) -> dict[str, Any]:
    """
    Convert a JudgeResult to an OTel span event dict.

    Usage
    -----
    >>> span.add_event(**judge_to_span_event(judge_result))
    """
    return result.to_otel_event()


def fingerprint_to_span_attributes(fp) -> dict[str, Any]:
    """
    Attach behavioral fingerprint signals as OTel span attributes.
    Useful for filtering/grouping traces in any backend by behavioral signal.

    Usage
    -----
    >>> span.set_attributes(fingerprint_to_span_attributes(fp))
    """
    return {
        "telemetry_eval.effort_to_outcome_ratio": fp.effort_to_outcome_ratio,
        "telemetry_eval.retry_density": fp.retry_density,
        "telemetry_eval.span_depth_mean": fp.span_depth_mean,
        "telemetry_eval.tool_dispatch_entropy": fp.tool_dispatch_entropy,
        "telemetry_eval.error_rate": fp.error_rate,
        "telemetry_eval.tool_call_count": fp.tool_call_count,
        "telemetry_eval.task_count": fp.task_count,
    }
