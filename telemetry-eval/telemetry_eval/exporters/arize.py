"""
Arize Phoenix Exporter
======================
Writes behavioral drift scores and ROPOLL judge results to Arize Phoenix
as span evaluations via the OpenInference / OTel span convention.

Requires: pip install arize-phoenix-otel openinference-instrumentation

Arize Phoenix natively consumes OTel spans. This exporter adds
telemetry-eval scores as span attributes and evaluation records that
appear in the Phoenix UI "Evaluations" tab.

Usage
-----
    from telemetry_eval.exporters.arize import ArizePhoenixExporter
    from telemetry_eval import DriftDetector, extract_fingerprints

    exporter = ArizePhoenixExporter()

    baseline_fps  = extract_fingerprints(baseline_traces)
    candidate_fps = extract_fingerprints(candidate_traces)
    drift_result  = DriftDetector().detect(baseline_fps, candidate_fps)

    # Attach drift as span attributes on the candidate span
    exporter.drift_attributes(drift_result)
    # → dict ready for span.set_attributes(...)

    # Or emit as a Phoenix-compatible evaluation event
    span.add_event(**exporter.drift_to_eval_event(drift_result))

Phoenix also accepts scores via its REST Evaluations API.
Use drift_result.to_otel_event() directly for pure OTel pipelines.
"""

from __future__ import annotations

from typing import Any

from telemetry_eval.core.drift import DriftResult
from telemetry_eval.judges.ropoll import JudgeResult


# Phoenix / OpenInference evaluation attribute prefix
_EVAL_PREFIX = "eval"


class ArizePhoenixExporter:
    """
    Thin adapter: formats telemetry-eval results for Arize Phoenix.

    Phoenix is an OTel-native backend — this class produces attribute
    dicts and event dicts in the format Phoenix expects for evaluations.
    No Phoenix SDK import required unless you use post_evaluations().

    Parameters
    ----------
    eval_name : str
        Name shown in the Phoenix Evaluations tab. Defaults to
        "behavioral_drift" / "ropoll_judge".
    """

    def __init__(self, eval_name: str | None = None):
        self.drift_eval_name = eval_name or "behavioral_drift"
        self.judge_eval_name = eval_name or "ropoll_judge"

    # ------------------------------------------------------------------
    # Span attributes (set directly on a span)
    # ------------------------------------------------------------------

    def drift_attributes(self, result: DriftResult) -> dict[str, Any]:
        """
        Returns a dict of span attributes for a DriftResult.
        Pass to span.set_attributes() in any OTel-instrumented code.

        Follows OpenInference evaluation attribute convention so Phoenix
        renders this under the Evaluations tab automatically.
        """
        prefix = f"{_EVAL_PREFIX}.{self.drift_eval_name}"
        attrs: dict[str, Any] = {
            f"{prefix}.label": result.label,
            f"{prefix}.score": result.score,
            f"{prefix}.explanation": result.comment,
            "telemetry_eval.ks_statistic": result.composite_ks_statistic,
            "telemetry_eval.p_value": result.min_p_value,
            "telemetry_eval.ci_lower": result.ci_lower,
            "telemetry_eval.ci_upper": result.ci_upper,
        }
        # Per-signal breakdown
        for sig in result.signals:
            key = f"telemetry_eval.signal.{sig.signal_name}"
            attrs[f"{key}.ks"] = sig.ks_statistic
            attrs[f"{key}.p_value"] = sig.p_value
            attrs[f"{key}.drifted"] = sig.drifted
            attrs[f"{key}.delta_pct"] = sig.delta_pct
        return attrs

    def judge_attributes(self, result: JudgeResult) -> dict[str, Any]:
        """
        Returns a dict of span attributes for a JudgeResult.
        Follows OpenInference evaluation attribute convention.
        """
        prefix = f"{_EVAL_PREFIX}.{self.judge_eval_name}"
        return {
            f"{prefix}.score": result.robust_score,
            f"{prefix}.explanation": result.comment,
            "telemetry_eval.judge.naive_mean": result.naive_mean,
            "telemetry_eval.judge.std": result.score_std,
            "telemetry_eval.judge.n_samples": result.n_samples,
            "telemetry_eval.judge.drift_detected": result.judge_drift_detected,
        }

    # ------------------------------------------------------------------
    # OTel span events  (add_event API)
    # ------------------------------------------------------------------

    def drift_to_eval_event(self, result: DriftResult) -> dict[str, Any]:
        """
        Returns an OTel span event dict for Phoenix's evaluation pipeline.
        Usage: span.add_event(**exporter.drift_to_eval_event(result))
        """
        return result.to_otel_event()

    def judge_to_eval_event(self, result: JudgeResult) -> dict[str, Any]:
        """
        Returns an OTel span event dict for Phoenix's evaluation pipeline.
        Usage: span.add_event(**exporter.judge_to_eval_event(result))
        """
        return result.to_otel_event()

    # ------------------------------------------------------------------
    # Phoenix REST Evaluations API  (optional, requires phoenix client)
    # ------------------------------------------------------------------

    def post_evaluations(
        self,
        client: Any,
        span_id: str,
        drift_result: DriftResult | None = None,
        judge_result: JudgeResult | None = None,
    ) -> None:
        """
        Post evaluations to Phoenix via its REST client.

        Requires: pip install arize-phoenix
        client : phoenix.Client  —  your existing Phoenix client.

        Parameters
        ----------
        client : phoenix.Client
        span_id : str
            The OTel span ID to attach evaluations to.
        drift_result : DriftResult, optional
        judge_result : JudgeResult, optional

        Example
        -------
        >>> import phoenix as px
        >>> client = px.Client()
        >>> exporter.post_evaluations(client, span_id, drift_result=result)
        """
        import pandas as pd  # noqa: F401 — required by Phoenix eval API

        if drift_result is not None:
            client.log_evaluations(
                SpanEvaluations(
                    eval_name=self.drift_eval_name,
                    dataframe=_make_eval_df(
                        span_id,
                        label=drift_result.label,
                        score=drift_result.score,
                        explanation=drift_result.comment,
                    ),
                )
            )

        if judge_result is not None:
            client.log_evaluations(
                SpanEvaluations(
                    eval_name=self.judge_eval_name,
                    dataframe=_make_eval_df(
                        span_id,
                        label="JUDGE_DRIFT" if judge_result.judge_drift_detected else "OK",
                        score=judge_result.robust_score,
                        explanation=judge_result.comment,
                    ),
                )
            )


def _make_eval_df(span_id: str, label: str, score: float, explanation: str) -> "pd.DataFrame":
    """Build a minimal Phoenix SpanEvaluations DataFrame."""
    import pandas as pd
    return pd.DataFrame(
        [{"context.span_id": span_id, "label": label, "score": score, "explanation": explanation}]
    ).set_index("context.span_id")


# Late import alias — only needed when post_evaluations() is called
try:
    from phoenix.evals import SpanEvaluations  # type: ignore
except ImportError:
    SpanEvaluations = None  # type: ignore
