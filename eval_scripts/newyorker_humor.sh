#!/usr/bin/env bash
# ============================================================================
# Evaluate: New Yorker Humor Understanding
# Dataset ID: newyorker_humor
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2209.06293
# Repo:  https://huggingface.co/datasets/jmhessel/newyorker_caption_contest

# ============================================================================
#
# Usage:
#   ./newyorker_humor.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./newyorker_humor.sh openai/gpt-4o
#   ./newyorker_humor.sh openai/gpt-4o my-suite
#   ./newyorker_humor.sh openai/gpt-4o my-suite 50
#   NEWYORKER_HUMOR_TASK=ranking ./newyorker_humor.sh openai/gpt-4o my-suite 10
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
TASK="${NEWYORKER_HUMOR_TASK:-matching}"

# ── Run entries ─────────────────────────────────────────────────────────────
case "$TASK" in
    matching)
        RUN_ENTRY="newyorker_humor:model=${MODEL}"
        TASK_LABEL="matching"
        ;;
    ranking)
        RUN_ENTRY="newyorker_humor_ranking:model=${MODEL}"
        TASK_LABEL="ranking"
        ;;
    *)
        echo "Error: NEWYORKER_HUMOR_TASK must be 'matching' or 'ranking' (got: $TASK)" >&2
        exit 1
        ;;
esac

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  New Yorker Humor Understanding"
echo "  Task:     $TASK_LABEL"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
