#!/usr/bin/env bash
# ============================================================================
# Evaluate: The Next Chapter
# Dataset ID: thenextchapter
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2301.09790
# Repo:  https://github.com/ZhuohanX/TheNextChapter

# ============================================================================
#
# Usage:
#   ./thenextchapter.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./thenextchapter.sh openai/gpt-4o
#   ./thenextchapter.sh openai/gpt-4o my-suite
#   ./thenextchapter.sh openai/gpt-4o my-suite 50
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
SUBSET="${THENEXTCHAPTER_SUBSET:-roc}"

python - <<'PY'
import nltk

for resource in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
PY

# ── Run entries ─────────────────────────────────────────────────────────────
RUN_ENTRIES=()
case "$SUBSET" in
  roc)
    RUN_ENTRIES+=("thenextchapter:model=${MODEL}")
    ;;
  wp)
    RUN_ENTRIES+=("thenextchapter_wp:model=${MODEL}")
    ;;
  cnn)
    RUN_ENTRIES+=("thenextchapter_cnn:model=${MODEL}")
    ;;
  all)
    RUN_ENTRIES+=("thenextchapter:model=${MODEL}")
    RUN_ENTRIES+=("thenextchapter_wp:model=${MODEL}")
    RUN_ENTRIES+=("thenextchapter_cnn:model=${MODEL}")
    ;;
  *)
    echo "Error: THENEXTCHAPTER_SUBSET must be one of roc, wp, cnn, all" >&2
    exit 1
    ;;
esac

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "${RUN_ENTRIES[@]}" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  The Next Chapter"
echo "  Subset:   $SUBSET"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""


# AGC_JUDGE_OVERRIDE pass-through (auto-inserted)
# AGC_JUDGE_OVERRIDE (global) takes priority over per-bench override; both are
# read by run_specs/thenextchapter_run_specs.py via llm_judge._judge_override.resolve_judge.
export AGC_JUDGE_OVERRIDE="${AGC_JUDGE_OVERRIDE:-}"
export THENEXTCHAPTER_JUDGE_MODEL_OVERRIDE="${THENEXTCHAPTER_JUDGE_MODEL_OVERRIDE:-}"

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
# disabled for parallel sweep: helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
