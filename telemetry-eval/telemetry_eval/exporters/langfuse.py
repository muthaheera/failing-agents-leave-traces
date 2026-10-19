"""
Langfuse Exporter
=================
Writes behavioral drift scores and ROPOLL judge results back to Langfuse
via the Scores API and Dataset API.

Requires: pip install langfuse

Langfuse stores scores on traces — once written, drift and judge scores
appear in the Langfuse UI alongside your existing traces automatically.
No Langfuse config changes needed.

Usage
-----
    from telemetry_eval.exporters.langfuse import LangfuseExporter
    from telemetry_eval import DriftDetector, extract_fingerprints

    exporter = LangfuseExporter(langfuse_client)

    baseline_fps  = extract_fingerprints(baseline_traces)
    candidate_fps = extract_fingerprints(candidate_traces)
    drift_result  = DriftDetector().detect(baseline_fps, candidate_fps)

    # Write drift score to every candidate trace in Langfuse
    for trace_id in candidate_trace_ids:
        exporter.write_drift_score(trace_id, drift_result)

    # Write ROPOLL judge score
    judge_result = ropoll_judge.evaluate(input_text, output_text)
    exporter.write_judge_score(trace_id, judge_result)
"""

from __future__ import annotations

from typing import Any

from telemetry_eval.core.drift import DriftResult
from telemetry_eval.judges.ropoll import JudgeResult


class LangfuseExporter:
    """
    Thin adapter: writes telemetry-eval results to Langfuse as scores.

    Parameters
    ----------
    client : langfuse.Langfuse
        Your existing initialised Langfuse client.
        telemetry-eval does NOT import langfuse at module level —
        only when this class is used — so langfuse is an optional dependency.
    """

    def __init__(self, client: Any):
        self.client = client

    def write_drift_score(self, trace_id: str, result: DriftResult) -> None:
        """
        POST drift result as a Langfuse score on the given trace.

        The score appears in the Langfuse UI trace view under "Scores"
        as "behavioral_drift" with value 1.0 (no drift) or 0.0 (drift).

        Parameters
        ----------
        trace_id : str
            Langfuse trace ID to attach the score to.
        result : DriftResult
            From DriftDetector.detect().
        """
        payload = result.to_langfuse_score()
        self.client.score(
            trace_id=trace_id,
            name=payload["name"],
            value=payload["value"],
            comment=payload["comment"],
            data_type=payload["dataType"],
        )

    def write_judge_score(self, trace_id: str, result: JudgeResult) -> None:
        """
        POST ROPOLL judge result as a Langfuse score on the given trace.

        Parameters
        ----------
        trace_id : str
            Langfuse trace ID to attach the score to.
        result : JudgeResult
            From ROPOLLJudge.evaluate().
        """
        payload = result.to_langfuse_score()
        self.client.score(
            trace_id=trace_id,
            name=payload["name"],
            value=payload["value"],
            comment=payload["comment"],
            data_type=payload["dataType"],
        )

    def write_fingerprint_scores(self, trace_id: str, fp) -> None:
        """
        Write each behavioral signal as a separate Langfuse score.
        Enables per-signal trend dashboards in Langfuse.

        Parameters
        ----------
        trace_id : str
        fp : BehavioralFingerprint
        """
        signals = {
            "effort_to_outcome_ratio": fp.effort_to_outcome_ratio,
            "retry_density": fp.retry_density,
            "span_depth_mean": fp.span_depth_mean,
            "tool_dispatch_entropy": fp.tool_dispatch_entropy,
            "error_rate": fp.error_rate,
        }
        for name, value in signals.items():
            self.client.score(
                trace_id=trace_id,
                name=f"behavioral_{name}",
                value=value,
                data_type="NUMERIC",
            )
