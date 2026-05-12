#!/usr/bin/env bash
# ============================================================================
# Evaluate: Rebus Puzzle
# Dataset ID: rebus_puzzle
# Input: image, text -> Output: text
# Paper: https://arxiv.org/abs/2505.23759
# Repo:  https://github.com/Kyunnilee/visual_puzzles

# ============================================================================
#
# Usage:
#   ./rebus_puzzle.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./rebus_puzzle.sh openai/gpt-4o
#   ./rebus_puzzle.sh openai/gpt-4o my-suite
#   ./rebus_puzzle.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="rebus_puzzle:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Rebus Puzzle"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

if [ -n "$MAX_INSTANCES" ]; then
    export REBUS_PUZZLE_INSTANCE_LIMIT_HINT="$MAX_INSTANCES"
fi

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
