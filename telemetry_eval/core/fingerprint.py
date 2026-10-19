"""
Behavioral Fingerprint Extractor
=================================
Computes behavioral execution signatures from OpenTelemetry agent spans.

Core finding from the paper:
  Behavioral effort-to-outcome ratio and tool dispatch patterns predict agent
  quality degradation with p < 0.001 (KS test), catching failures that
  LLM-as-a-Judge scores missed entirely (judge: 4.48/5 while 2x degradation
  was already underway).

Paper:
  "Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation"
  Muthaheera Yasmeena — REALM Workshop, EMNLP 2026
  https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation

Input contract
--------------
Accepts standard OpenTelemetry span dicts (as exported by any OTel backend):
  {
    "trace_id":    str,
    "span_id":     str,
    "parent_span_id": str | None,
    "name":        str,          # e.g. "execute_tool", "invoke_agent"
    "start_time":  float,        # Unix epoch seconds
    "end_time":    float,
    "status":      "OK" | "ERROR" | "UNSET",
    "attributes":  dict,         # gen_ai.* attributes per OTel GenAI spec
    "events":      list[dict],   # span events
  }

This format is emitted natively by:
  - Langfuse (export via SDK or REST)
  - Arize Phoenix (OTel-native)
  - LangSmith (OTel export added 2025)
  - OpenLLMetry / Traceloop
  - Any backend that ingests OTLP
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BehavioralFingerprint:
    """
    A compact behavioral signature for a single agent trace.

    All five signals are derived from execution telemetry alone —
    no model output content is inspected.

    Fields
    ------
    trace_id : str
        Identifies the source trace.
    effort_to_outcome_ratio : float
        Total tool calls / tasks completed.
        Rising ratio signals the agent is working harder for the same result —
        the primary drift indicator from the paper (Section 4.2).
    retry_density : float
        Retried tool calls / total tool calls.
        Captures silent retry loops invisible to LLM judges.
    span_depth_mean : float
        Mean nesting depth of execution spans.
        Deeper trees indicate more complex (or confused) planning.
    tool_dispatch_entropy : float
        Shannon entropy of the tool-call distribution.
        Low entropy → agent stuck calling the same tool repeatedly.
        High entropy spike → erratic, unplanned tool usage.
    error_rate : float
        Spans with status=ERROR / total spans.
    duration_seconds : float
        Wall-clock duration of the root span.
    task_count : int
        Number of completed task spans detected.
    tool_call_count : int
        Total tool invocations in this trace.
    """
    trace_id: str
    effort_to_outcome_ratio: float
    retry_density: float
    span_depth_mean: float
    tool_dispatch_entropy: float
    error_rate: float
    duration_seconds: float
    task_count: int
    tool_call_count: int
    raw_attributes: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signal_vector(self) -> list[float]:
        """Returns the five core signals as a numeric vector for KS/bootstrap tests."""
        return [
            self.effort_to_outcome_ratio,
            self.retry_density,
            self.span_depth_mean,
            self.tool_dispatch_entropy,
            self.error_rate,
        ]


# ---------------------------------------------------------------------------
# Span classification helpers
# ---------------------------------------------------------------------------

# OTel GenAI semantic convention span names (v1.41 spec)
_TOOL_SPAN_NAMES = {"execute_tool", "tool_call", "tool", "function_call"}
_TASK_SPAN_NAMES = {"invoke_agent", "create_agent", "invoke_workflow", "agent", "task"}
_RETRY_ATTR_KEYS = {"retry", "retry_count", "attempt", "gen_ai.retry_count"}


def _is_tool_span(span: dict) -> bool:
    name = span.get("name", "").lower()
    return any(t in name for t in _TOOL_SPAN_NAMES)


def _is_task_span(span: dict) -> bool:
    name = span.get("name", "").lower()
    return any(t in name for t in _TASK_SPAN_NAMES)


def _is_retry(span: dict) -> bool:
    attrs = span.get("attributes", {})
    return any(k in attrs and int(attrs[k]) > 0 for k in _RETRY_ATTR_KEYS if k in attrs)


def _span_depth(span_id: str, parent_map: dict[str, str | None]) -> int:
    depth = 0
    current = span_id
    visited = set()
    while current in parent_map and parent_map[current] is not None:
        if current in visited:
            break
        visited.add(current)
        current = parent_map[current]
        depth += 1
    return depth


def _shannon_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_fingerprint(spans: list[dict[str, Any]]) -> BehavioralFingerprint:
    """
    Extract a BehavioralFingerprint from a list of OTel span dicts.

    Works with raw span exports from Langfuse, Arize Phoenix, LangSmith,
    or any OTLP-compatible backend.

    Parameters
    ----------
    spans : list[dict]
        All spans belonging to a single trace. Order does not matter.

    Returns
    -------
    BehavioralFingerprint

    Example
    -------
    >>> import json
    >>> spans = json.load(open("my_trace.json"))   # exported from Langfuse / Phoenix
    >>> fp = extract_fingerprint(spans)
    >>> print(fp.effort_to_outcome_ratio)
    """
    if not spans:
        return BehavioralFingerprint(
            trace_id="", effort_to_outcome_ratio=0.0, retry_density=0.0,
            span_depth_mean=0.0, tool_dispatch_entropy=0.0, error_rate=0.0,
            duration_seconds=0.0, task_count=0, tool_call_count=0,
        )

    # Build parent map for depth calculation
    parent_map: dict[str, str | None] = {
        s["span_id"]: s.get("parent_span_id") for s in spans
    }

    # Find root span (no parent)
    root_span = next(
        (s for s in spans if not s.get("parent_span_id")),
        spans[0]
    )
    trace_id = root_span.get("trace_id", root_span.get("span_id", "unknown"))
    duration = root_span.get("end_time", 0) - root_span.get("start_time", 0)

    # Classify spans
    tool_spans = [s for s in spans if _is_tool_span(s)]
    task_spans = [s for s in spans if _is_task_span(s)]
    error_spans = [s for s in spans if s.get("status") == "ERROR"]
    retry_spans = [s for s in tool_spans if _is_retry(s)]

    tool_call_count = len(tool_spans)
    task_count = max(len(task_spans), 1)  # avoid div/0; at least 1 task per trace

    # Signal 1: Effort-to-outcome ratio (Section 4.2 of paper)
    effort_to_outcome = tool_call_count / task_count

    # Signal 2: Retry density
    retry_density = len(retry_spans) / tool_call_count if tool_call_count > 0 else 0.0

    # Signal 3: Span depth mean
    depths = [_span_depth(s["span_id"], parent_map) for s in spans]
    span_depth_mean = sum(depths) / len(depths) if depths else 0.0

    # Signal 4: Tool dispatch entropy — which tools are being called and how often
    tool_name_counts: dict[str, int] = {}
    for s in tool_spans:
        tool_name = (
            s.get("attributes", {}).get("gen_ai.tool.name")
            or s.get("attributes", {}).get("tool.name")
            or s.get("name", "unknown")
        )
        tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
    tool_dispatch_entropy = _shannon_entropy(tool_name_counts)

    # Signal 5: Error rate
    error_rate = len(error_spans) / len(spans) if spans else 0.0

    # Collect raw attributes from root for downstream exporters
    raw_attributes = root_span.get("attributes", {})

    return BehavioralFingerprint(
        trace_id=trace_id,
        effort_to_outcome_ratio=effort_to_outcome,
        retry_density=retry_density,
        span_depth_mean=span_depth_mean,
        tool_dispatch_entropy=tool_dispatch_entropy,
        error_rate=error_rate,
        duration_seconds=duration,
        task_count=task_count,
        tool_call_count=tool_call_count,
        raw_attributes=raw_attributes,
    )


def extract_fingerprints(traces: list[list[dict[str, Any]]]) -> list[BehavioralFingerprint]:
    """
    Batch extraction across multiple traces.

    Parameters
    ----------
    traces : list of span lists
        Each inner list is all spans for one trace.
    """
    return [extract_fingerprint(spans) for spans in traces]
