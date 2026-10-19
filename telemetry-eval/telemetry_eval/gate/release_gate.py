"""
Release Gate — Statistically-Gated CI/CD Block
===============================================
Blocks a release if behavioral drift is detected with statistical significance.

Solves the "flaky eval" problem: standard CI gates fail because LLM outputs
are non-deterministic (temperature, GPU floating-point variance, dynamic
batching). A naive pass/fail threshold triggers false alarms.

This gate only blocks when drift is STATISTICALLY SIGNIFICANT (p < threshold)
— not on random noise.

Usage
-----
From Python:
    from telemetry_eval import ReleaseGate, extract_fingerprints

    gate = ReleaseGate()
    verdict = gate.evaluate(baseline_spans, candidate_spans)
    if not verdict.passed:
        raise SystemExit(verdict.summary)

From CLI (for GitHub Actions / any CI):
    python -m telemetry_eval.gate \\
        --baseline baseline_traces.json \\
        --candidate candidate_traces.json \\
        --p-threshold 0.001

    # Exits 0 (pass) or 1 (block) — standard CI convention

Paper:
  "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
  Muthaheera Yasmeena — REALM Workshop, EMNLP 2026
  https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from telemetry_eval.core.fingerprint import extract_fingerprints, BehavioralFingerprint
from telemetry_eval.core.drift import DriftDetector, DriftResult


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class GateVerdict:
    """
    Release gate decision.

    passed=True  → release is safe to proceed
    passed=False → release is BLOCKED due to behavioral drift
    """
    passed: bool
    drift_result: DriftResult
    reason: str

    @property
    def exit_code(self) -> int:
        """0 = pass, 1 = block. Standard Unix/CI convention."""
        return 0 if self.passed else 1

    @property
    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "🚫 BLOCK"
        return f"{status} | {self.reason}"

    def to_github_output(self) -> str:
        """
        Formats result for GitHub Actions $GITHUB_OUTPUT.
        Add to your workflow step to propagate gate result.
        """
        lines = [
            f"gate_passed={'true' if self.passed else 'false'}",
            f"gate_label={self.drift_result.label}",
            f"gate_score={self.drift_result.score}",
            f"gate_p_value={self.drift_result.min_p_value:.6f}",
            f"gate_ks_stat={self.drift_result.composite_ks_statistic:.4f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "drift": self.drift_result.to_dict(),
        }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class ReleaseGate:
    """
    Statistically-gated release block using behavioral drift detection.

    Parameters
    ----------
    p_threshold : float
        KS test significance threshold. Default 0.001 (production-grade).
        Use 0.05 for earlier, more sensitive warnings in development.
    n_bootstrap : int
        Bootstrap resamples for CI estimation.
    min_traces : int
        Minimum traces required in each set. Gate is skipped (passed=True
        with a warning) if either set is too small for reliable statistics.

    Example — GitHub Actions
    ------------------------
    # In your CI workflow:
    - name: Behavioral drift gate
      run: |
        python -m telemetry_eval.gate \\
          --baseline baseline.json \\
          --candidate candidate.json \\
          --p-threshold 0.001
      # Exits 1 and fails the step if drift detected
    """

    def __init__(
        self,
        p_threshold: float = 0.001,
        n_bootstrap: int = 1000,
        min_traces: int = 10,
    ):
        self.p_threshold = p_threshold
        self.n_bootstrap = n_bootstrap
        self.min_traces = min_traces
        self._detector = DriftDetector(
            p_threshold=p_threshold,
            n_bootstrap=n_bootstrap,
        )

    def evaluate(
        self,
        baseline_traces: list[list[dict]],
        candidate_traces: list[list[dict]],
    ) -> GateVerdict:
        """
        Run gate evaluation from raw span lists.

        Parameters
        ----------
        baseline_traces : list of span lists
            Each inner list = all spans for one trace (from baseline release).
        candidate_traces : list of span lists
            Each inner list = all spans for one trace (candidate release).

        Returns
        -------
        GateVerdict
        """
        # Guard: need enough traces for reliable statistics
        if len(baseline_traces) < self.min_traces or len(candidate_traces) < self.min_traces:
            reason = (
                f"Insufficient traces for statistical test "
                f"(baseline={len(baseline_traces)}, candidate={len(candidate_traces)}, "
                f"min={self.min_traces}). Gate skipped."
            )
            # Build a dummy DriftResult for the verdict
            from telemetry_eval.core.drift import DriftResult, SignalDrift, SIGNAL_NAMES
            dummy = DriftResult(
                drifted=False, signals=[], composite_ks_statistic=0.0,
                min_p_value=1.0, ci_lower=0.0, ci_upper=0.0,
                n_baseline=len(baseline_traces), n_candidate=len(candidate_traces),
                p_threshold=self.p_threshold,
            )
            return GateVerdict(passed=True, drift_result=dummy, reason=reason)

        baseline_fps = extract_fingerprints(baseline_traces)
        candidate_fps = extract_fingerprints(candidate_traces)
        return self.evaluate_fingerprints(baseline_fps, candidate_fps)

    def evaluate_fingerprints(
        self,
        baseline: list[BehavioralFingerprint],
        candidate: list[BehavioralFingerprint],
    ) -> GateVerdict:
        """
        Run gate evaluation from pre-computed fingerprints.
        Use this if you already extracted fingerprints upstream.
        """
        result = self._detector.detect(baseline, candidate)

        if result.drifted:
            drifted_signals = [s.signal_name for s in result.signals if s.drifted]
            reason = (
                f"Behavioral drift detected in: {', '.join(drifted_signals)}. "
                f"KS={result.composite_ks_statistic:.3f}, "
                f"p={result.min_p_value:.4f} < {self.p_threshold}. "
                f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]. "
                f"Release BLOCKED."
            )
        else:
            reason = (
                f"No significant behavioral drift. "
                f"min p={result.min_p_value:.4f} >= {self.p_threshold}. "
                f"Release approved."
            )

        return GateVerdict(
            passed=not result.drifted,
            drift_result=result,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# CLI entrypoint  (python -m telemetry_eval.gate ...)
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Behavioral drift release gate.\n"
            "Exits 0 (pass) or 1 (block).\n\n"
            "Paper: https://www.amazon.science/publications/"
            "failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to JSON file: list of trace span lists (baseline release).",
    )
    parser.add_argument(
        "--candidate", required=True,
        help="Path to JSON file: list of trace span lists (candidate release).",
    )
    parser.add_argument(
        "--p-threshold", type=float, default=0.001,
        help="KS test p-value threshold. Default: 0.001 (production). Use 0.05 for dev.",
    )
    parser.add_argument(
        "--min-traces", type=int, default=10,
        help="Minimum traces required per set. Default: 10.",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Optional path to write full JSON result.",
    )
    parser.add_argument(
        "--github-output", action="store_true",
        help="Print GitHub Actions output format to stdout.",
    )
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline_traces = json.load(f)
    with open(args.candidate) as f:
        candidate_traces = json.load(f)

    gate = ReleaseGate(p_threshold=args.p_threshold, min_traces=args.min_traces)
    verdict = gate.evaluate(baseline_traces, candidate_traces)

    print(verdict.summary)

    if args.github_output:
        print(verdict.to_github_output())

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(verdict.to_dict(), f, indent=2)
        print(f"Full result written to {args.output_json}")

    sys.exit(verdict.exit_code)


if __name__ == "__main__":
    _cli()
