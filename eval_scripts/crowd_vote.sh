#!/usr/bin/env bash
# ============================================================================
# Evaluate: CrowdVote
# Dataset ID: crowd_vote
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2509.09702
# Repo:  (not available)

# ============================================================================
#
# Usage:
#   ./crowd_vote.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./crowd_vote.sh openai/gpt-4o
#   ./crowd_vote.sh openai/gpt-4o my-suite
#   ./crowd_vote.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="crowd_vote:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  CrowdVote"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""


# AGC_JUDGE_OVERRIDE pass-through (auto-inserted)
# AGC_JUDGE_OVERRIDE (global) takes priority over per-bench override; both are
# read by run_specs/crowd_vote_run_specs.py via llm_judge._judge_override.resolve_judge.
export AGC_JUDGE_OVERRIDE="${AGC_JUDGE_OVERRIDE:-}"
export CROWD_VOTE_JUDGE_MODEL_OVERRIDE="${CROWD_VOTE_JUDGE_MODEL_OVERRIDE:-}"

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
