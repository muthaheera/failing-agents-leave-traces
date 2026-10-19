"""
LangSmith Exporter
==================
Writes behavioral drift scores and ROPOLL judge results to LangSmith
as run feedback via the LangSmith SDK.

Requires: pip install langsmith

LangSmith stores feedback on runs. Once written, drift and judge scores
appear in the LangSmith UI under "Feedback" for any run.

Usage
-----
    from telemetry_eval.exporters.langsmith import LangSmithExporter
    from telemetry_eval import DriftDetector, extract_fingerprints

    exporter = LangSmithExporter(langsmith_client)

    baseline_fps  = extract_fingerprints(baseline_traces)
    candidate_fps = extract_fingerprints(candidate_traces)
    drift_result  = DriftDetector().detect(baseline_fps, candidate_fps)

    # Write drift score as feedback on a LangSmith run
    for run_id in candidate_run_ids:
        exporter.write_drift_feedback(run_id, drift_result)

    # Write ROPOLL judge result
    judge_result = ropoll_judge.evaluate(input_text, output_text)
    exporter.write_judge_feedback(run_id, judge_result)

No LangSmith config changes needed — scores appear automatically
in the Feedback column and aggregation charts.
"""

from __future__ import annotations

from typing import Any

from telemetry_eval.core.drift import DriftResult
from telemetry_eval.judges.ropoll import JudgeResult


class LangSmithExporter:
    """
    Thin adapter: writes telemetry-eval results to LangSmith as run feedback.

    Parameters
    ----------
    client : langsmith.Client
        Your existing initialised LangSmith client.
        telemetry-eval does NOT import langsmith at module level —
        only when this class is used — so langsmith is an optional dependency.
    source_info : dict, optional
        Extra metadata to include in the feedback source dict.
        Useful for tracking which release / experiment the feedback came from.
    """

    def __init__(self, client: Any, source_info: dict | None = None):
        self.client = client
        self.source_info = source_info or {}

    def write_drift_feedback(self, run_id: str, result: DriftResult) -> None:
        """
        Write drift result as LangSmith run feedback.

        Creates a "behavioral_drift" feedback score (0.0 or 1.0) on the run.
        Appears in the LangSmith UI Feedback column and aggregate charts.

        Parameters
        ----------
        run_id : str
            LangSmith run ID to attach feedback to.
        result : DriftResult
            From DriftDetector.detect().
        """
        self.client.create_feedback(
            run_id=run_id,
            key="behavioral_drift",
            score=result.score,
            value=result.label,
            comment=result.comment,
            source_info={
                **self.source_info,
                "ks_statistic": result.composite_ks_statistic,
                "p_value": result.min_p_value,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "n_baseline": result.n_baseline,
                "n_candidate": result.n_candidate,
                "paper": (
                    "https://www.amazon.science/publications/"
                    "failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation"
                ),
            },
        )

    def write_judge_feedback(self, run_id: str, result: JudgeResult) -> None:
        """
        Write ROPOLL judge result as LangSmith run feedback.

        Creates a "ropoll_judge" feedback score on the run. If judge drift
        was detected (naive mean diverged from robust score), a second
        "judge_naive_mean" score is also written for comparison.

        Parameters
        ----------
        run_id : str
            LangSmith run ID to attach feedback to.
        result : JudgeResult
            From ROPOLLJudge.evaluate().
        """
        self.client.create_feedback(
            run_id=run_id,
            key="ropoll_judge",
            score=result.robust_score,
            comment=result.comment,
            source_info={
                **self.source_info,
                "n_samples": result.n_samples,
                "naive_mean": result.naive_mean,
                "score_std": result.score_std,
                "judge_drift_detected": result.judge_drift_detected,
                "judge_drift_delta": result.judge_drift_delta,
            },
        )

        # Write naive mean separately so teams can see the divergence in dashboards
        if result.judge_drift_detected:
            self.client.create_feedback(
                run_id=run_id,
                key="judge_naive_mean",
                score=result.naive_mean,
                comment=f"Naive mean (ROPOLL robust={result.robust_score:.3f}, delta={result.judge_drift_delta:.3f})",
                source_info=self.source_info,
            )

    def write_fingerprint_feedback(self, run_id: str, fp) -> None:
        """
        Write each behavioral fingerprint signal as a separate LangSmith feedback.
        Enables per-signal trend dashboards in LangSmith analytics.

        Parameters
        ----------
        run_id : str
        fp : BehavioralFingerprint
        """
        signals = {
            "behavioral_effort_to_outcome_ratio": fp.effort_to_outcome_ratio,
            "behavioral_retry_density": fp.retry_density,
            "behavioral_span_depth_mean": fp.span_depth_mean,
            "behavioral_tool_dispatch_entropy": fp.tool_dispatch_entropy,
            "behavioral_error_rate": fp.error_rate,
        }
        for key, value in signals.items():
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=value,
                source_info=self.source_info,
            )
