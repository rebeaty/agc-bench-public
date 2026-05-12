#!/usr/bin/env bash
# ============================================================================
# Evaluate: Future Ideas
# Dataset ID: future_ideas
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2409.06185
# Repo:  https://github.com/sandeep82945/Future-Idea-Generation

# ============================================================================
#
# Usage:
#   ./future_ideas.sh MODEL [SUITE] [MAX_INSTANCES] [DOMAIN]
#
# Examples:
#   ./future_ideas.sh openai/gpt-4o
#   ./future_ideas.sh openai/gpt-4o my-suite
#   ./future_ideas.sh openai/gpt-4o my-suite 50
#   ./future_ideas.sh openai/gpt-4o my-suite 10 physics
#
# Arguments:
#   MODEL          Required. The model to evaluate (e.g., openai/gpt-4o).
#   SUITE          Optional. Name for this evaluation run (default: agc-bench).
#   MAX_INSTANCES  Optional. Limit the number of test instances (useful for quick tests).
#   DOMAIN         Optional. One of: all, chemistry, computer, economics, medical, physics.
#                  Default: all.
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES] [DOMAIN]}"
SUITE="${2:-agc-bench}"
MAX_INSTANCES="${3:-}"
DOMAIN="${4:-all}"

# ── Run entries ─────────────────────────────────────────────────────────────
case "$DOMAIN" in
    all) RUN_SPEC="future_ideas" ;;
    chemistry|computer|economics|medical|physics) RUN_SPEC="future_ideas_${DOMAIN}" ;;
    *)
        echo "Error: DOMAIN must be one of: all, chemistry, computer, economics, medical, physics" >&2
        exit 1
        ;;
esac

RUN_ENTRY="${RUN_SPEC}:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Future Ideas"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "  Domain:   $DOMAIN"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""


# AGC_JUDGE_OVERRIDE pass-through (auto-inserted)
# AGC_JUDGE_OVERRIDE (global) takes priority over per-bench override; both are
# read by run_specs/future_ideas_run_specs.py via llm_judge._judge_override.resolve_judge.
export AGC_JUDGE_OVERRIDE="${AGC_JUDGE_OVERRIDE:-}"
export FUTURE_IDEAS_JUDGE_MODEL_OVERRIDE="${FUTURE_IDEAS_JUDGE_MODEL_OVERRIDE:-}"

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
