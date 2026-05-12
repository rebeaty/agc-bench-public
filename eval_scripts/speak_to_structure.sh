#!/usr/bin/env bash
# ============================================================================
# Evaluate: Speak-to-Structure (S2-Bench / TOMG-Bench)
# Dataset ID: speak_to_structure
# Input: text -> Output: text
# Paper: https://openreview.net/forum?id=qTTmUJFG38
# Repo:  https://github.com/phenixace/S2-TOMG-Bench

# ============================================================================
#
# Usage:
#   ./speak_to_structure.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./speak_to_structure.sh openai/gpt-4o
#   ./speak_to_structure.sh openai/gpt-4o my-suite
#   ./speak_to_structure.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="speak_to_structure:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Speak-to-Structure (S2-Bench / TOMG-Bench)"
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
