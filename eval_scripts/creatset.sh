#!/usr/bin/env bash
# ============================================================================
# Evaluate: CrEval / CreataSet
# Dataset ID: creatset
# Input: Chinese instruction -> Generated response
# Paper: https://arxiv.org/abs/2505.19236
# Repo:  https://github.com/Aman-4-Real/CrEval

# ============================================================================
#
# Usage:
#   ./creatset.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./creatset.sh openai/gpt-4o
#   ./creatset.sh openai/gpt-4o my-suite
#   ./creatset.sh openai/gpt-4o my-suite 50
#
# Arguments:
#   MODEL          Required. The model to evaluate (e.g., openai/gpt-4o).
#   SUITE          Optional. Name for this evaluation run (default: agc-bench).
#   MAX_INSTANCES  Optional. Limit the number of test instances (useful for quick tests).
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-agc-bench}"
MAX_INSTANCES="${3:-}"

# ── Run entries ─────────────────────────────────────────────────────────────
RUN_ENTRY="creatset:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  CrEval / CreataSet"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""


# AGC_JUDGE_OVERRIDE pass-through (auto-inserted)
# AGC_JUDGE_OVERRIDE (global) takes priority over per-bench override; both are
# read by run_specs/creatset_run_specs.py via llm_judge._judge_override.resolve_judge.
export AGC_JUDGE_OVERRIDE="${AGC_JUDGE_OVERRIDE:-}"
export CREATSET_JUDGE_MODEL_OVERRIDE="${CREATSET_JUDGE_MODEL_OVERRIDE:-}"

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
