# telemetry-eval

**Behavioral evaluation layer for multi-tier AI agents.**
Integration endpoint for Langfuse · Arize Phoenix · LangSmith · any OTel backend.

> **Paper:** [Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation](https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation)
> Muthaheera Yasmeena — REALM Workshop, EMNLP 2026, Budapest

---

## Why this exists

LLM judges miss silent degradation. In the paper, a 2× system degradation was scored **4.48/5.00** by LLM judges — missed entirely. OTel spans caught it at **p=0.0485**.

`telemetry-eval` adds a behavioral layer *underneath* your existing eval stack.
It extracts 5 signals from OTel spans, runs a KS test across releases, and writes drift scores directly into Langfuse, Arize Phoenix, or LangSmith — **no new dashboards, no new workflow**.

---

## The 3-line integration

```python
from telemetry_eval import DriftDetector, extract_fingerprints

baseline_fps  = extract_fingerprints(baseline_traces)   # OTel spans you already have
candidate_fps = extract_fingerprints(candidate_traces)
drift_result  = DriftDetector().detect(baseline_fps, candidate_fps)
```

The result goes straight to your existing observability backend:

```python
# Langfuse
from telemetry_eval.exporters.langfuse import LangfuseExporter
LangfuseExporter(langfuse_client).write_drift_score(trace_id, drift_result)

# Arize Phoenix
from telemetry_eval.exporters.arize import ArizePhoenixExporter
span.set_attributes(ArizePhoenixExporter().drift_attributes(drift_result))

# LangSmith
from telemetry_eval.exporters.langsmith import LangSmithExporter
LangSmithExporter(langsmith_client).write_drift_feedback(run_id, drift_result)

# Pure OTel (any backend)
from telemetry_eval.exporters.otel import drift_to_span_event
span.add_event(**drift_to_span_event(drift_result))
```

---

## Install

```bash
pip install telemetry-eval                # core: OTel + KS test
pip install "telemetry-eval[langfuse]"    # + Langfuse
pip install "telemetry-eval[arize]"       # + Arize Phoenix
pip install "telemetry-eval[langsmith]"   # + LangSmith
pip install "telemetry-eval[all]"         # everything
```

**Requirements:** Python ≥ 3.9, scipy ≥ 1.9, opentelemetry-api ≥ 1.20.  
No vendor SDK required for the core — only installed if you use that exporter.

---

## What it measures

`telemetry-eval` extracts **5 behavioral signals** from standard OTel spans:

| Signal | What it captures | Degradation signal |
|--------|------------------|--------------------|
| `effort_to_outcome_ratio` | Span duration per output token | Agent working harder for less |
| `retry_density` | Retry spans / total spans | Tool call instability |
| `span_depth_mean` | Mean depth of span tree | Unnecessary orchestration nesting |
| `tool_dispatch_entropy` | Shannon entropy of tool call distribution | Abnormal tool selection patterns |
| `error_rate` | Error spans / total spans | Direct failure signal |

These are the **Behavioral Fingerprint** from the paper (Section 3).

---

## Drift detection

```python
from telemetry_eval import DriftDetector

detector = DriftDetector(
    p_threshold=0.001,   # production-grade: p < 0.001
    n_bootstrap=1000,    # bootstrap resamples for 95% CI
)
result = detector.detect(baseline_fps, candidate_fps)

print(result.label)                       # "DRIFT_DETECTED" or "NO_DRIFT"
print(result.score)                       # 0.0 (drift) or 1.0 (no drift)
print(result.composite_ks_statistic)      # max KS across 5 signals
print(result.min_p_value)                 # most significant signal
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

for sig in result.signals:
    print(f"{sig.signal_name}: KS={sig.ks_statistic:.3f}, Δ={sig.delta_pct:+.1f}%")
```

**Why KS test?** The paper shows LLM output distributions are non-Gaussian. KS is non-parametric — it makes no distributional assumptions and has breakdown point 0 (detects any distributional shift).

**Why bootstrap CI?** LLM eval is noisy. A 95% CI on the composite KS statistic distinguishes real degradation from sampling variance — solving the "flaky eval" problem in CI/CD.

---

## ROPOLL: Robust LLM judge aggregation

ROPOLL replaces naive averaging of LLM judge scores with geometric median aggregation (breakdown point 1/2: up to half the judges can be biased without corrupting the result).

Based on: Acharya, Pan & Verkhovsky — ICML 2026 Workshop.

```python
from telemetry_eval import ROPOLLJudge

def my_judge(input_text: str, output_text: str) -> float:
    # Your existing OpenAI / Anthropic judge call
    return call_llm_judge(input_text, output_text)

judge = ROPOLLJudge(judge_fn=my_judge, n_samples=5)
result = judge.evaluate(input_text="...", output_text="...")

print(result.robust_score)          # geometric median — use this
print(result.naive_mean)            # what you'd get with averaging
print(result.judge_drift_detected)  # True if naive would have misled you
```

---

## Release gate (CI/CD)

Block releases when behavioral drift is detected with statistical significance.

**Python:**
```python
from telemetry_eval import ReleaseGate

gate = ReleaseGate(p_threshold=0.001)
verdict = gate.evaluate(baseline_traces, candidate_traces)

if not verdict.passed:
    raise SystemExit(verdict.summary)
```

**CLI:**
```bash
python -m telemetry_eval.gate \
  --baseline baseline_traces.json \
  --candidate candidate_traces.json \
  --p-threshold 0.001 \
  --github-output

# Exits 0 (pass) or 1 (block) — standard CI convention
```

**GitHub Actions:**
```yaml
- name: Behavioral drift gate
  run: |
    python -m telemetry_eval.gate \
      --baseline eval/baseline.json \
      --candidate eval/candidate.json
  # Fails the step and blocks the release if drift detected
```

See [`.github/workflows/eval-gate.yml`](.github/workflows/eval-gate.yml) for the full reusable workflow.

---

## Span format

`telemetry-eval` accepts standard OTel span dicts — the format any instrumentation library (OpenLLMetry, opentelemetry-python, Langfuse SDK) already produces:

```python
span = {
    "trace_id": "...",
    "span_id": "...",
    "parent_span_id": "...",      # None for root spans
    "name": "invoke_agent",       # or "execute_tool", "invoke_agent.retry", ...
    "start_time": 1700000000000,  # nanoseconds
    "end_time":   1700000001000,
    "status": {"status_code": "OK"},   # or "ERROR"
    "attributes": {
        "gen_ai.usage.output_tokens": 120,
        # ... any other attributes
    },
    "events": [],
}
```

**Span name conventions** (from OTel GenAI semantic conventions):
- `invoke_agent` — agent task span
- `execute_tool` — tool call span
- `*.retry` or `*.retry.*` — retry classification

Compatible natively with: Langfuse, Arize Phoenix, LangSmith, OpenLLMetry, any OTLP receiver.

---

## Examples

See [`examples/`](examples/) for working quickstarts:

- [`examples/otel_quickstart.py`](examples/otel_quickstart.py) — pure OTel, works with any backend
- [`examples/langfuse_quickstart.py`](examples/langfuse_quickstart.py) — Langfuse Scores API integration

---

## Architecture

```
Your agent (any LLM framework)
        │ OTel spans
        ▼
┌─────────────────────────────────────────────┐
│            telemetry-eval                   │
│                                             │
│  extract_fingerprints(traces)               │
│  → BehavioralFingerprint × N               │
│       (5 signals per trace)                 │
│                                             │
│  DriftDetector.detect(baseline, candidate)  │
│  → DriftResult (KS + bootstrap CI)         │
│                                             │
│  ReleaseGate.evaluate(...)                  │
│  → GateVerdict (pass/block + exit code)    │
│                                             │
│  ROPOLLJudge.evaluate(input, output)        │
│  → JudgeResult (robust score)              │
└─────────────┬───────────────────────────────┘
              │ thin exporters (20 lines each)
    ┌─────────┼──────────┬──────────┐
    ▼         ▼          ▼          ▼
 Langfuse  Arize      LangSmith  Any OTel
 (Scores)  (Phoenix   (Feedback) backend
           Evals)
```

**Design principle:** `telemetry-eval` is not an observability platform. It is the behavioral evaluation layer *underneath* eval that makes all other eval frameworks more honest.

---

## Paper and citation

```bibtex
@inproceedings{yasmeena2026telemetry,
  title     = {Failing Agents Leave Traces: Telemetry Signatures for Multi-Tier Agent Evaluation},
  author    = {Yasmeena, Muthaheera},
  booktitle = {Proceedings of the REALM Workshop at EMNLP 2026},
  year      = {2026},
  url       = {https://www.amazon.science/publications/failing-agents-leave-traces-telemetry-signatures-for-multi-tier-agent-evaluation}
}
```

**Key finding:** KS drift monitors (p < 0.001) detected a 2× system degradation that LLM judges scored 4.48/5.00 — missed entirely by judge-only evaluation pipelines.

---

## Contributing

Issues and PRs welcome. This is an integration endpoint, not a standalone platform — the goal is to make behavioral drift scores show up in the tools teams already use.

## License

Apache 2.0
