#!/usr/bin/env bash
# ============================================================================
# Evaluate: Conceptual Design
# Dataset ID: conceptual_design
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2306.01779
# Repo:  https://github.com/kevinma1515/gpt_idetc

# ============================================================================
#
# Usage:
#   ./conceptual_design.sh MODEL [SUITE] [MAX_INSTANCES] [PROMPT_VARIANT]
#
# Examples:
#   ./conceptual_design.sh openai/gpt-4o
#   ./conceptual_design.sh openai/gpt-4o my-suite
#   ./conceptual_design.sh openai/gpt-4o my-suite 50
#
# Arguments:
#   MODEL          Required. The model to evaluate (e.g., openai/gpt-4o).
#   SUITE          Optional. Name for this evaluation run (default: agc-bench).
#   MAX_INSTANCES  Optional. Limit the number of test instances (useful for quick tests).
#   PROMPT_VARIANT Optional. One of base, novel, diverse, unique (default: base).
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-agc-bench}"
MAX_INSTANCES="${3:-}"
PROMPT_VARIANT="${4:-base}"

# ── Run entries ─────────────────────────────────────────────────────────────
RUN_ENTRY="conceptual_design:model=${MODEL},prompt_variant=${PROMPT_VARIANT}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Conceptual Design"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "  Prompt variant: $PROMPT_VARIANT"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""


# AGC_JUDGE_OVERRIDE pass-through (auto-inserted)
# AGC_JUDGE_OVERRIDE (global) takes priority over per-bench override; both are
# read by run_specs/conceptual_design_run_specs.py via llm_judge._judge_override.resolve_judge.
export AGC_JUDGE_OVERRIDE="${AGC_JUDGE_OVERRIDE:-}"
export CONCEPTUAL_DESIGN_JUDGE_MODEL_OVERRIDE="${CONCEPTUAL_DESIGN_JUDGE_MODEL_OVERRIDE:-}"

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
