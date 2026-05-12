#!/usr/bin/env bash
# ============================================================================
# Evaluate: FutureGen
# Dataset ID: futuregen
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2503.16561
# Repo:  https://github.com/IbrahimAlAzhar/FutureWorkGeneration

# ============================================================================
#
# Usage:
#   ./futuregen.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./futuregen.sh openai/gpt-4o
#   ./futuregen.sh openai/gpt-4o my-suite
#   ./futuregen.sh openai/gpt-4o my-suite 50
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
PROMPT_STYLE="${FUTUREGEN_PROMPT_STYLE:-top3}"

# ── Run entries ─────────────────────────────────────────────────────────────
case "$PROMPT_STYLE" in
    top3)
        RUN_ENTRY="futuregen:model=${MODEL}"
        ;;
    all_sections)
        RUN_ENTRY="futuregen_all_sections:model=${MODEL}"
        ;;
    *)
        echo "Error: FUTUREGEN_PROMPT_STYLE must be 'top3' or 'all_sections' (got '$PROMPT_STYLE')." >&2
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
echo "  Dataset:  FutureGen"
echo "  Prompt:   $PROMPT_STYLE"
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
