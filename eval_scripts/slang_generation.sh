#!/usr/bin/env bash
# ============================================================================
# Evaluate: Slang Generation
# Dataset ID: slang_generation
# Input: text -> Output: text
# Paper: https://aclanthology.org/2025.findings-emnlp.348/
# Repo:  https://github.com/siyangwu1/LLM-Slang-Dictionary

# ============================================================================
#
# Usage:
#   ./slang_generation.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./slang_generation.sh openai/gpt-4o
#   ./slang_generation.sh openai/gpt-4o my-suite
#   ./slang_generation.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="slang_generation:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Slang Generation"
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
