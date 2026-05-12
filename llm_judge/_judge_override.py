"""Shared judge-model override helper.

Every LLM-judge run_spec calls `resolve_judge(default, bench_var=None)` instead
of hard-coding a judge model identifier. The resolver checks two env vars in priority order:

  1. AGC_JUDGE_OVERRIDE — global flag that swaps every of the 24 LLM-judge cells
     to the same judge (used by `eval_scripts/run_with_agc_judge.sh` to route
     all judge calls through AGC-Judge).
  2. <BENCH>_JUDGE_MODEL_OVERRIDE — per-bench override (legacy; useful when
     swapping a single bench, e.g. for fidelity-audit re-runs).
  3. The run_spec's hard-coded paper-canonical default.

Returns a string suitable for HELM's AnnotatorSpec `judge_model_name` arg —
i.e. a HELM model-deployment identifier. The identifier must resolve through HELM's
AutoClient, which means either:

  - It's a built-in HELM-supported identifier (e.g. `google/gemini-3-flash-preview`,
    `anthropic/claude-3.7-sonnet`, `openai/gpt-5.4-mini`)
  - It's registered in `prod_env/model_deployments.yaml` (e.g.
    `agcbench-2026/agc-judge`).

If you set AGC_JUDGE_OVERRIDE to an identifier HELM doesn't know, the run will fail
at HELM dispatch time with a clear "no model deployment" error — fix it by
adding the deployment to `prod_env/model_deployments.yaml` per the HELM docs
on adding new models.
"""
from __future__ import annotations
import os
from typing import Optional


def resolve_judge(default: str, bench_var: Optional[str] = None) -> str:
    """Resolve the judge-model identifier for an AnnotatorSpec.

    Args:
        default: The paper-canonical judge identifier (used if no override is set).
        bench_var: Optional per-bench env-var name (e.g.
            "ARASTORIES_JUDGE_MODEL_OVERRIDE"). If provided, takes precedence
            over `default` but is itself overridden by AGC_JUDGE_OVERRIDE.

    Returns:
        The judge identifier to pass to AnnotatorSpec(args={"judge_model_name": ...}).
    """
    g = os.environ.get("AGC_JUDGE_OVERRIDE", "").strip()
    if g:
        return g
    if bench_var:
        b = os.environ.get(bench_var, "").strip()
        if b:
            return b
    return default
