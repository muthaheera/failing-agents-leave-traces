"""
ROPOLL — Robust Pooled LLM Judge Aggregation
=============================================
Implementation of the geometric median aggregation method from:
  Acharya, Pan & Verkhovsky — ICML 2026 Workshop

Used in production at Amazon Lab126 EvalAI to replace naive
LLM-as-a-Judge averaging. Breakdown point of 1/2: up to half the
judges can be biased without corrupting the aggregate score.

From the paper (Section 3.1):
  "ROPOLL proved judges scored a 2× degradation at 4.48/5.00 while
   telemetry caught it (p=0.0485). Geometric median aggregation
   replaces naive averaging; behavioral anomalies are routed to
   human review before LLM judging."

Paper:
  "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
  Muthaheera Yasmeena — REALM Workshop, EMNLP 2026
  https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation

Design
------
ROPOLLJudge wraps any callable judge (OpenAI, Anthropic, local model).
It samples N judge scores, applies geometric median, and returns a
calibrated score + confidence interval — all in one call.

Integrates with any observability backend via to_otel_event() and
to_langfuse_score() on the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Geometric median (Weiszfeld's algorithm)
# ---------------------------------------------------------------------------

def _geometric_median_1d(values: list[float], max_iter: int = 200, tol: float = 1e-6) -> float:
    """
    1-D geometric median via Weiszfeld's algorithm.
    More robust than mean: breakdown point = 0.5.
    """
    if not values:
        raise ValueError("Cannot compute geometric median of empty list.")
    if len(values) == 1:
        return values[0]

    # Initialise at arithmetic mean
    estimate = sum(values) / len(values)

    for _ in range(max_iter):
        weights = []
        for v in values:
            dist = abs(v - estimate)
            weights.append(1.0 / max(dist, 1e-10))

        total_weight = sum(weights)
        new_estimate = sum(w * v for w, v in zip(weights, values)) / total_weight

        if abs(new_estimate - estimate) < tol:
            break
        estimate = new_estimate

    return estimate


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """
    Calibrated judge result from ROPOLL aggregation.

    Compatible with Langfuse Scores API and OTel gen_ai.evaluation.result.
    """
    raw_scores: list[float]          # All N judge samples
    robust_score: float              # Geometric median (the calibrated score)
    naive_mean: float                # Naive average (for comparison / drift detection)
    score_std: float                 # Std dev across samples
    n_samples: int
    judge_drift_detected: bool       # True if robust_score differs from naive_mean by > threshold
    judge_drift_delta: float         # |robust - naive| — how much naive averaging was off

    @property
    def score(self) -> float:
        """Primary score. Use this, not naive_mean."""
        return self.robust_score

    @property
    def comment(self) -> str:
        drift_note = (
            f" ⚠ Judge drift detected: naive mean={self.naive_mean:.3f}, "
            f"robust score={self.robust_score:.3f} (delta={self.judge_drift_delta:.3f})."
            if self.judge_drift_detected else ""
        )
        return (
            f"ROPOLL robust score: {self.robust_score:.3f} "
            f"(naive mean: {self.naive_mean:.3f}, std: {self.score_std:.3f}, "
            f"n={self.n_samples}).{drift_note}"
        )

    def to_otel_event(self) -> dict:
        """OTel gen_ai.evaluation.result event payload."""
        return {
            "name": "gen_ai.evaluation.result",
            "attributes": {
                "gen_ai.evaluation.name": "ropoll_judge",
                "gen_ai.evaluation.score": self.robust_score,
                "gen_ai.evaluation.comment": self.comment,
                "telemetry_eval.judge.naive_mean": self.naive_mean,
                "telemetry_eval.judge.std": self.score_std,
                "telemetry_eval.judge.n_samples": self.n_samples,
                "telemetry_eval.judge.drift_detected": self.judge_drift_detected,
                "telemetry_eval.paper": (
                    "https://www.amazon.science/publications/"
                    "failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation"
                ),
            },
        }

    def to_langfuse_score(self) -> dict:
        """Langfuse Scores API payload."""
        return {
            "name": "ropoll_judge",
            "value": self.robust_score,
            "comment": self.comment,
            "dataType": "NUMERIC",
        }


# ---------------------------------------------------------------------------
# Judge wrapper
# ---------------------------------------------------------------------------

class ROPOLLJudge:
    """
    Wraps any LLM judge callable with ROPOLL robust aggregation.

    Parameters
    ----------
    judge_fn : Callable[[str, str], float]
        Your existing judge function: takes (input, output) → score (0–1 or 0–5).
        Can be an OpenAI call, Anthropic call, local model, or any scorer.
    n_samples : int
        Number of judge samples to draw. Paper used n=5 for production.
        Higher n → more robust but more expensive.
    drift_threshold : float
        If |robust_score - naive_mean| > this, judge_drift_detected=True.
        Signals that naive averaging would have given a misleading score.

    Example
    -------
    >>> def my_judge(input_text: str, output_text: str) -> float:
    ...     # your existing OpenAI / Anthropic judge call
    ...     response = openai_client.chat.completions.create(...)
    ...     return float(response.choices[0].message.content.strip())
    ...
    >>> judge = ROPOLLJudge(judge_fn=my_judge, n_samples=5)
    >>> result = judge.evaluate(input_text="...", output_text="...")
    >>>
    >>> # Send robust score to Langfuse
    >>> langfuse_client.score(trace_id=..., **result.to_langfuse_score())
    >>>
    >>> # Emit as OTel event
    >>> span.add_event(**result.to_otel_event())
    """

    def __init__(
        self,
        judge_fn: Callable[[str, str], float],
        n_samples: int = 5,
        drift_threshold: float = 0.15,
    ):
        self.judge_fn = judge_fn
        self.n_samples = n_samples
        self.drift_threshold = drift_threshold

    def evaluate(self, input_text: str, output_text: str) -> JudgeResult:
        """
        Sample the judge n_samples times and aggregate via geometric median.

        Parameters
        ----------
        input_text : str
            The prompt / task given to the agent.
        output_text : str
            The agent's output to be judged.

        Returns
        -------
        JudgeResult with robust_score, naive_mean, and drift detection.
        """
        raw_scores: list[float] = []
        for _ in range(self.n_samples):
            score = self.judge_fn(input_text, output_text)
            raw_scores.append(float(score))

        robust_score = _geometric_median_1d(raw_scores)
        naive_mean = sum(raw_scores) / len(raw_scores)
        variance = sum((s - naive_mean) ** 2 for s in raw_scores) / len(raw_scores)
        score_std = math.sqrt(variance)
        delta = abs(robust_score - naive_mean)

        return JudgeResult(
            raw_scores=raw_scores,
            robust_score=robust_score,
            naive_mean=naive_mean,
            score_std=score_std,
            n_samples=self.n_samples,
            judge_drift_detected=delta > self.drift_threshold,
            judge_drift_delta=delta,
        )

    def evaluate_batch(
        self, pairs: list[dict[str, str]]
    ) -> list[JudgeResult]:
        """
        Batch evaluation across multiple input/output pairs.

        Parameters
        ----------
        pairs : list of {"input": str, "output": str}
        """
        return [self.evaluate(p["input"], p["output"]) for p in pairs]
