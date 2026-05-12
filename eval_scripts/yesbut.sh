#!/usr/bin/env bash
# ============================================================================
# Evaluate: YesBut
# Dataset ID: yesbut
# Input: image -> Output: text
# Paper: https://arxiv.org/abs/2409.13592
# Repo:  https://huggingface.co/datasets/bansalaman18/yesbut

# ============================================================================
#
# Usage:
#   ./yesbut.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./yesbut.sh openai/gpt-4o
#   ./yesbut.sh openai/gpt-4o my-suite
#   ./yesbut.sh openai/gpt-4o my-suite 50
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
DIFFICULTY="${YESBUT_DIFFICULTY:-all}"

# ── Run entries ─────────────────────────────────────────────────────────────
RUN_ENTRY="yesbut:model=${MODEL}"

# Ensure NLTK METEOR resources are available for the paper-aligned metric bundle.
python - <<'PY'
import nltk

for resource in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
PY

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  YesBut (Understanding / WHYFUNNY)"
echo "  Difficulty: $DIFFICULTY"
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
