#!/usr/bin/env bash
# ============================================================================
# Evaluate: Simile Generation (SCOPE)
# Dataset ID: simile_generation
# Input: text -> Output: text
# Paper: https://aclanthology.org/2020.emnlp-main.524/
# Repo:  https://github.com/tuhinjubcse/SimileGeneration-EMNLP2020

# ============================================================================
#
# Usage:
#   ./simile_generation.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./simile_generation.sh openai/gpt-4o
#   ./simile_generation.sh openai/gpt-4o my-suite
#   ./simile_generation.sh openai/gpt-4o my-suite 50
#
# Arguments:
#   MODEL          Required. The model to evaluate (e.g., openai/gpt-4o).
#   SUITE          Optional. Name for this evaluation run (default: agc-bench).
#   MAX_INSTANCES  Optional. Limit the number of test instances (useful for quick tests).
#
# Environment:
#   SIMILE_GENERATION_BERTSCORE_MODEL_OVERRIDE
#                  Optional. Override the BERTScore backbone used by the
#                  benchmark-specific automatic metric. Defaults to
#                  `roberta-large` to match the released scorer's `lang="en"`
#                  behavior.
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-agc-bench}"
MAX_INSTANCES="${3:-}"

# ── Run entries ─────────────────────────────────────────────────────────────
RUN_ENTRY="simile_generation:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Simile Generation (SCOPE)"
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
