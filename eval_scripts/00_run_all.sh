#!/usr/bin/env bash
# ============================================================================
# Sequential runner for all AGC-Bench dataset evaluation scripts.
#
# Prefer eval_scripts/00_run_all_parallel.sh for full benchmark runs. This
# sequential runner is kept for debugging one dataset after another while
# preserving the same per-dataset wrapper behavior.
#
# Usage:
#   ./eval_scripts/00_run_all.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Example:
#   ./eval_scripts/00_run_all.sh openai/gpt-4o agc-bench-smoke 10
# ============================================================================

set -euo pipefail

MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-agc-bench}"
MAX_INSTANCES="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASSED=0
FAILED=0
FAILED_LIST=()

for script in "$SCRIPT_DIR"/*.sh; do
    name="$(basename "$script" .sh)"
    case "$name" in
        00_run_all|00_run_all_parallel|run_all|_helm_run|run_with_agc_judge|serve_agc_judge_locally) continue ;;
    esac

    echo ""
    echo "================================================================"
    echo "  Running: $name"
    echo "================================================================"

    if "$script" "$MODEL" "$SUITE" "$MAX_INSTANCES"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$name")
        echo "WARNING: $name failed, continuing..."
    fi
done

echo ""
echo "================================================================"
echo "  Summary: $PASSED passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo "  Failed: ${FAILED_LIST[*]}"
fi
echo "================================================================"

# Final summarize
# disabled for parallel sweep: helm-summarize --suite "$SUITE"
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
