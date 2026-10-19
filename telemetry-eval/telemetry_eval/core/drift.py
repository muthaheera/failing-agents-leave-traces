"""
Drift Detector — Kolmogorov-Smirnov + Bootstrap Confidence Intervals
=====================================================================
Detects behavioral distribution shift between a baseline and a candidate
set of agent traces.

From the paper (Section 4.3):
  "Non-parametric KS drift monitors (p < 0.001) serve as the production
   release gate for distribution shift detection. The KS test detected a
   2× system degradation that LLM judges scored 4.48/5.00 — missed entirely
   by judge-only evaluation pipelines."

Paper:
  "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
  Muthaheera Yasmeena — REALM Workshop, EMNLP 2026
  https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation

Design principle
----------------
Output is a structured DriftResult that any observability backend can consume:
  - Write to Langfuse as a score via their Scores API
  - Emit as gen_ai.evaluation.result OTel event
  - Post to a webhook (GitHub Actions, PagerDuty, Slack)
  - Return as JSON from a REST gate endpoint
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from scipy import stats  # type: ignore

from telemetry_eval.core.fingerprint import BehavioralFingerprint

# Signal names matching BehavioralFingerprint.signal_vector() order
SIGNAL_NAMES = [
    "effort_to_outcome_ratio",
    "retry_density",
    "span_depth_mean",
    "tool_dispatch_entropy",
    "error_rate",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SignalDrift:
    """KS test result for a single behavioral signal."""
    signal_name: str
    ks_statistic: float
    p_value: float
    drifted: bool           # True if p_value < threshold
    baseline_mean: float
    candidate_mean: float
    delta_pct: float        # % change from baseline mean


@dataclass
class DriftResult:
    """
    Complete drift analysis result for a candidate release.

    Compatible with:
      - OTel gen_ai.evaluation.result event (score + label)
      - Langfuse Scores API (name, value, comment)
      - Arize Phoenix span evaluations
      - JSON webhook payloads
    """
    drifted: bool                       # Overall verdict
    signals: list[SignalDrift]          # Per-signal breakdown
    composite_ks_statistic: float       # Max KS across all signals
    min_p_value: float                  # Most significant signal
    ci_lower: float                     # Bootstrap 95% CI lower bound (composite score)
    ci_upper: float                     # Bootstrap 95% CI upper bound
    n_baseline: int
    n_candidate: int
    p_threshold: float                  # Threshold used for this evaluation

    # --- OTel gen_ai.evaluation.result compatible fields ---
    @property
    def score(self) -> float:
        """1.0 = no drift detected, 0.0 = significant drift. For OTel/Langfuse."""
        return 0.0 if self.drifted else 1.0

    @property
    def label(self) -> Literal["DRIFT_DETECTED", "NO_DRIFT"]:
        return "DRIFT_DETECTED" if self.drifted else "NO_DRIFT"

    @property
    def comment(self) -> str:
        drifted_signals = [s.signal_name for s in self.signals if s.drifted]
        if self.drifted:
            return (
                f"Behavioral drift detected (p={self.min_p_value:.4f} < {self.p_threshold}). "
                f"Drifted signals: {', '.join(drifted_signals)}. "
                f"Composite KS={self.composite_ks_statistic:.3f}. "
                f"95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]."
            )
        return (
            f"No significant drift (min p={self.min_p_value:.4f} >= {self.p_threshold}). "
            f"95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]."
        )

    def to_otel_event(self) -> dict:
        """
        Formats result as an OTel gen_ai.evaluation.result event.
        Drop this dict into any OTLP span event payload.
        """
        return {
            "name": "gen_ai.evaluation.result",
            "attributes": {
                "gen_ai.evaluation.name": "behavioral_drift",
                "gen_ai.evaluation.score": self.score,
                "gen_ai.evaluation.result": self.label,
                "gen_ai.evaluation.comment": self.comment,
                "telemetry_eval.ks_statistic": self.composite_ks_statistic,
                "telemetry_eval.p_value": self.min_p_value,
                "telemetry_eval.ci_lower": self.ci_lower,
                "telemetry_eval.ci_upper": self.ci_upper,
                "telemetry_eval.paper": (
                    "https://www.amazon.science/publications/"
                    "failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation"
                ),
            },
        }

    def to_langfuse_score(self) -> dict:
        """
        Formats result for Langfuse Scores API.
        POST to /api/public/scores
        """
        return {
            "name": "behavioral_drift",
            "value": self.score,
            "comment": self.comment,
            "dataType": "NUMERIC",
        }

    def to_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "label": self.label,
            "score": self.score,
            "composite_ks_statistic": self.composite_ks_statistic,
            "min_p_value": self.min_p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_baseline": self.n_baseline,
            "n_candidate": self.n_candidate,
            "p_threshold": self.p_threshold,
            "signals": [
                {
                    "signal": s.signal_name,
                    "ks": s.ks_statistic,
                    "p_value": s.p_value,
                    "drifted": s.drifted,
                    "baseline_mean": s.baseline_mean,
                    "candidate_mean": s.candidate_mean,
                    "delta_pct": s.delta_pct,
                }
                for s in self.signals
            ],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Compares behavioral fingerprints between a baseline and a candidate release.

    Usage (works with any span source)
    ----
    >>> from telemetry_eval import DriftDetector, extract_fingerprints
    >>>
    >>> baseline_fps  = extract_fingerprints(baseline_traces)
    >>> candidate_fps = extract_fingerprints(candidate_traces)
    >>>
    >>> detector = DriftDetector(p_threshold=0.001, n_bootstrap=1000)
    >>> result = detector.detect(baseline_fps, candidate_fps)
    >>>
    >>> # Send to Langfuse
    >>> langfuse_client.score(trace_id=..., **result.to_langfuse_score())
    >>>
    >>> # Emit as OTel event
    >>> span.add_event(**result.to_otel_event())
    >>>
    >>> # Block CI/CD release
    >>> if result.drifted:
    ...     sys.exit(1)

    Parameters
    ----------
    p_threshold : float
        KS test p-value below which drift is declared.
        Paper used p < 0.001 for production gates.
        Use p < 0.05 for development / earlier warning.
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation.
        1000 is sufficient for 95% CI. Use 5000 for tighter bounds.
    random_seed : int | None
        Set for reproducible CI computation.
    """

    def __init__(
        self,
        p_threshold: float = 0.001,
        n_bootstrap: int = 1000,
        random_seed: int | None = 42,
    ):
        self.p_threshold = p_threshold
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed

    def detect(
        self,
        baseline: list[BehavioralFingerprint],
        candidate: list[BehavioralFingerprint],
    ) -> DriftResult:
        """
        Run KS test + bootstrap CI across all behavioral signals.

        Parameters
        ----------
        baseline : list[BehavioralFingerprint]
            Fingerprints from the known-good release.
        candidate : list[BehavioralFingerprint]
            Fingerprints from the release under evaluation.

        Returns
        -------
        DriftResult
            Structured result consumable by any observability backend.
        """
        if not baseline or not candidate:
            raise ValueError("Both baseline and candidate must be non-empty.")

        baseline_vecs = [fp.signal_vector() for fp in baseline]
        candidate_vecs = [fp.signal_vector() for fp in candidate]

        signal_results: list[SignalDrift] = []
        any_drift = False
        max_ks = 0.0
        min_p = 1.0

        for i, name in enumerate(SIGNAL_NAMES):
            b_vals = [v[i] for v in baseline_vecs]
            c_vals = [v[i] for v in candidate_vecs]

            ks_stat, p_val = stats.ks_2samp(b_vals, c_vals)
            drifted = p_val < self.p_threshold

            b_mean = sum(b_vals) / len(b_vals)
            c_mean = sum(c_vals) / len(c_vals)
            delta_pct = ((c_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0.0

            signal_results.append(SignalDrift(
                signal_name=name,
                ks_statistic=ks_stat,
                p_value=p_val,
                drifted=drifted,
                baseline_mean=b_mean,
                candidate_mean=c_mean,
                delta_pct=delta_pct,
            ))

            if drifted:
                any_drift = True
            if ks_stat > max_ks:
                max_ks = ks_stat
            if p_val < min_p:
                min_p = p_val

        # Bootstrap CI on composite KS statistic
        ci_lower, ci_upper = self._bootstrap_ci(baseline_vecs, candidate_vecs)

        return DriftResult(
            drifted=any_drift,
            signals=signal_results,
            composite_ks_statistic=max_ks,
            min_p_value=min_p,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_baseline=len(baseline),
            n_candidate=len(candidate),
            p_threshold=self.p_threshold,
        )

    def _bootstrap_ci(
        self,
        baseline_vecs: list[list[float]],
        candidate_vecs: list[list[float]],
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap resample to estimate CI on composite KS statistic."""
        rng = random.Random(self.random_seed)
        boot_stats: list[float] = []

        for _ in range(self.n_bootstrap):
            b_sample = [rng.choice(baseline_vecs) for _ in range(len(baseline_vecs))]
            c_sample = [rng.choice(candidate_vecs) for _ in range(len(candidate_vecs))]

            max_ks = 0.0
            for i in range(len(SIGNAL_NAMES)):
                b_vals = [v[i] for v in b_sample]
                c_vals = [v[i] for v in c_sample]
                ks_stat, _ = stats.ks_2samp(b_vals, c_vals)
                if ks_stat > max_ks:
                    max_ks = ks_stat
            boot_stats.append(max_ks)

        boot_stats.sort()
        alpha = 1 - confidence
        lo_idx = int(alpha / 2 * self.n_bootstrap)
        hi_idx = int((1 - alpha / 2) * self.n_bootstrap)
        return boot_stats[lo_idx], boot_stats[min(hi_idx, len(boot_stats) - 1)]
