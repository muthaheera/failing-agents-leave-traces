"""
telemetry-eval
==============
Behavioral evaluation layer for AI agents.

Plugs into any OTel-compatible observability backend
(Langfuse, Arize Phoenix, LangSmith, MLflow, custom).

Based on:
  "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
  Muthaheera Yasmeena Belgur Shamiullah — REALM Workshop, EMNLP 2026
  https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation
"""

from telemetry_eval.core.fingerprint import BehavioralFingerprint, extract_fingerprint
from telemetry_eval.core.drift import DriftDetector, DriftResult
from telemetry_eval.gate.release_gate import ReleaseGate, GateVerdict
from telemetry_eval.judges.ropoll import ROPOLLJudge

__version__ = "0.1.0"
__author__ = "Muthaheera Yasmeena Belgur Shamiullah"
__paper__ = "https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation"

__all__ = [
    "BehavioralFingerprint",
    "extract_fingerprint",
    "DriftDetector",
    "DriftResult",
    "ReleaseGate",
    "GateVerdict",
    "ROPOLLJudge",
]
