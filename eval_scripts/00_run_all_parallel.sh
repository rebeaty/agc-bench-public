#!/usr/bin/env bash
# ============================================================================
# Orchestrator: full AGC-Bench evaluation (parallel)
#
# Runs the per-dataset evaluation scripts in ./eval_scripts/ for every dataset
# in scenarios/. Datasets execute concurrently in batches of PARALLELISM. All
# runs share the HELM suite name "first_full_trial" and land in
# benchmark_output/runs/first_full_trial/.
#
# Usage:
#   ./eval_scripts/00_run_all_parallel.sh MODEL [MAX_INSTANCES] [PARALLELISM]
#
# Examples:
#   ./eval_scripts/00_run_all_parallel.sh google/gemini-2.5-flash-lite
#   ./eval_scripts/00_run_all_parallel.sh google/gemini-2.5-flash-lite -1 4
#   ./eval_scripts/00_run_all_parallel.sh google/gemini-2.5-flash-lite 10 8
#
# Arguments:
#   MODEL          Required. OpenRouter-formatted model identifier (e.g. vendor/model).
#   MAX_INSTANCES  Optional. Integer. -1 (default) means evaluate ALL instances.
#   PARALLELISM    Optional. How many dataset scripts to run concurrently. Default 4.
#
# Notes on model access:
#   - The evaluated MODEL is looked up on OpenRouter. If it is not listed, the
#     script exits with a warning BEFORE running any dataset.
#   - Some source-paper judge models are hard-coded inside dataset run_specs.
#     If AGC_JUDGE_OVERRIDE is set, those LLM-judge calls are routed to the
#     override instead and source-judge availability checks are skipped.
#   - Dataset downloads are handled inside the HELM scenario Python code, so
#     missing data is fetched automatically on first run.
# ============================================================================

set -uo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENARIOS_DIR="$PROJECT_ROOT/scenarios"
LOG_DIR="$PROJECT_ROOT/benchmark_output/runs/first_full_trial/_orchestrator_logs"
STATUS_DIR="$LOG_DIR/_status"

# ── Arguments ──────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [MAX_INSTANCES] [PARALLELISM]}"
MAX_INSTANCES="${2:--1}"
PARALLELISM="${3:-4}"
SUITE="first_full_trial"

if ! [[ "$PARALLELISM" =~ ^[0-9]+$ ]] || [ "$PARALLELISM" -lt 1 ]; then
    echo "ERROR: PARALLELISM must be a positive integer (got '$PARALLELISM')." >&2
    exit 2
fi

# Convert MAX_INSTANCES=-1 to empty so the downstream scripts run on all instances.
if [ "$MAX_INSTANCES" = "-1" ]; then
    MAX_ARG=""
else
    MAX_ARG="$MAX_INSTANCES"
fi

# ── Load .env if present (for OPENROUTER_API_KEY) ──────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ── Validate OpenRouter credentials ────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set. All model access (including judge" >&2
    echo "       models) is routed through OpenRouter. Aborting." >&2
    exit 2
fi

# ── OpenRouter model availability check ────────────────────────────────────
# OpenRouter does NOT expose a per-model GET (/api/v1/models/{id} returns 404
# for every model). Fetch the full list once, then check membership locally.
OPENROUTER_MODELS_CACHE="$(mktemp -t openrouter_models.XXXXXX.json)"
trap 'rm -f "$OPENROUTER_MODELS_CACHE"' EXIT

_openrouter_list_http=$(curl -s -o "$OPENROUTER_MODELS_CACHE" -w "%{http_code}" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    "https://openrouter.ai/api/v1/models")
if [ "$_openrouter_list_http" != "200" ]; then
    echo "ERROR: failed to fetch OpenRouter model list (HTTP $_openrouter_list_http)." >&2
    echo "       Check OPENROUTER_API_KEY and network access." >&2
    exit 2
fi

check_openrouter_model() {
    local model_id="$1"
    python3 - "$model_id" "$OPENROUTER_MODELS_CACHE" <<'PY'
import json, sys
model_id, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
ids = {m.get("id") for m in data.get("data", [])}
sys.exit(0 if model_id in ids else 1)
PY
}

echo "Verifying evaluation model is reachable on OpenRouter: $MODEL"
if ! check_openrouter_model "$MODEL"; then
    echo "WARNING: model '$MODEL' is not available on OpenRouter (HTTP != 200)." >&2
    echo "         Refusing to start trial. Use an OpenRouter-compatible model identifier" >&2
    echo "         like 'anthropic/claude-sonnet-4' or 'openai/gpt-4o'." >&2
    exit 3
fi
echo "  OK."

# Default source-paper judge models observed in run_specs/. These are checked
# only when AGC_JUDGE_OVERRIDE is not set. With AGC-Judge scoring, the override
# replaces these judge identifiers for the 24 LLM-judge benchmarks.
JUDGE_IDS=(
    "anthropic/claude-3.7-sonnet"
    "google/gemini-2.5-flash-lite"
    "google/gemini-3-flash-preview"
    "openai/gpt-4"
    "openai/gpt-4-turbo"
    "openai/gpt-4.1"
    "openai/gpt-4.1-mini"
    "openai/gpt-4o"
    "openai/gpt-4o-mini"
    "openai/o3-mini-2025-01-31"
)
if [[ -n "${AGC_JUDGE_OVERRIDE:-}" ]]; then
    echo "AGC_JUDGE_OVERRIDE is set (${AGC_JUDGE_OVERRIDE}); skipping source-judge checks."
else
    echo "Verifying OpenRouter-routed source judge models:"
    for judge_id in "${JUDGE_IDS[@]}"; do
        if [[ "$judge_id" == google/* ]]; then
            echo "  NOTE $judge_id uses the bundled Google GenAI deployment; ensure googleApiKey is configured."
            continue
        fi
        if check_openrouter_model "$judge_id"; then
            echo "  OK   $judge_id"
        else
            echo "WARNING: judge model '$judge_id' is not available on OpenRouter." >&2
            echo "         It is referenced by at least one dataset's run_spec." >&2
            echo "         Set AGC_JUDGE_OVERRIDE for AGC-Judge scoring, or update the" >&2
            echo "         corresponding per-benchmark judge override before running." >&2
            exit 4
        fi
    done
fi

# ── Load dataset list from scenarios/ ──────────────────────────────────────
if [ ! -d "$SCENARIOS_DIR" ]; then
    echo "ERROR: scenarios directory not found at $SCENARIOS_DIR" >&2
    exit 5
fi

# Discover datasets by scanning scenarios/*_scenario.py — strip the suffix.
mapfile -t ALL_SCENARIOS < <(
    find "$SCENARIOS_DIR" -maxdepth 1 -name "*_scenario.py" -type f \
        | xargs -n1 basename \
        | sed 's/_scenario\.py$//' \
        | sort
)

if [ "${#ALL_SCENARIOS[@]}" -eq 0 ]; then
    echo "ERROR: no scenarios found under $SCENARIOS_DIR" >&2
    exit 6
fi

# Default: restrict to the 67 primary datasets featured in the v1 paper
# (status=included in release_data/dataset_metadata.csv). Set
# AGC_INCLUDE_MULTIMODAL=1 to evaluate every scenario on disk (includes the
# 11 multimodal benches that are deferred to v1.1).
INCLUDE_CSV="$PROJECT_ROOT/release_data/dataset_metadata.csv"
if [[ "${AGC_INCLUDE_MULTIMODAL:-0}" == "1" ]] || [ ! -f "$INCLUDE_CSV" ]; then
    DATASETS=("${ALL_SCENARIOS[@]}")
    SCOPE_LABEL="all $(echo "${#DATASETS[@]}") scenarios on disk"
else
    mapfile -t INCLUDED < <(awk -F, 'NR>1 && $2=="included" {print $1}' "$INCLUDE_CSV" | sort -u)
    declare -A _included_set
    for d in "${INCLUDED[@]}"; do _included_set[$d]=1; done
    DATASETS=()
    for d in "${ALL_SCENARIOS[@]}"; do
        if [[ -n "${_included_set[$d]:-}" ]]; then
            DATASETS+=("$d")
        fi
    done
    SCOPE_LABEL="${#DATASETS[@]} primary datasets (status=included; v1 paper roster)"
fi

if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "ERROR: dataset filter produced an empty list" >&2
    exit 6
fi

mkdir -p "$LOG_DIR" "$STATUS_DIR"
# Clean stale status markers from a previous run.
rm -f "$STATUS_DIR"/*.status 2>/dev/null || true

# ── Single-dataset worker ──────────────────────────────────────────────────
# Runs one dataset script, writes a status marker when done. Safe to background.
run_one() {
    local idx="$1"
    local total="$2"
    local dataset="$3"
    local script="$SCRIPT_DIR/${dataset}.sh"
    local log="$LOG_DIR/${dataset}.log"
    local status="$STATUS_DIR/${dataset}.status"

    if [ ! -f "$script" ]; then
        printf 'SKIP\tno script at %s\n' "$script" >"$status"
        echo "[$idx/$total] SKIP $dataset (no script)"
        return 0
    fi

    echo "[$idx/$total] START $dataset"
    # Dataset data download is handled inside the HELM scenario's Python code
    # (get_instances -> ensure_file_downloaded / HuggingFace hub).
    if bash "$script" "$MODEL" "$SUITE" "$MAX_ARG" >"$log" 2>&1; then
        printf 'PASS\t0\n' >"$status"
        echo "[$idx/$total] PASS  $dataset"
    else
        local rc=$?
        printf 'FAIL\t%s\n' "$rc" >"$status"
        echo "[$idx/$total] FAIL  $dataset (rc=$rc, log: $log)" >&2
    fi
}

# ── Bounded parallel dispatch ──────────────────────────────────────────────
TOTAL=${#DATASETS[@]}

echo "================================================================"
echo "  AGC-Bench full evaluation"
echo "  Model:         $MODEL"
echo "  Suite:         $SUITE"
echo "  Max instances: ${MAX_ARG:-ALL}"
echo "  Parallelism:   $PARALLELISM"
echo "  Datasets:      $TOTAL  (${SCOPE_LABEL})"
echo "  Logs:          $LOG_DIR"
echo "================================================================"

running=0
i=0
for dataset in "${DATASETS[@]}"; do
    i=$((i + 1))

    # Throttle: if we already have PARALLELISM workers in flight, wait for one
    # to finish before launching the next.
    while [ "$running" -ge "$PARALLELISM" ]; do
        wait -n
        running=$((running - 1))
    done

    run_one "$i" "$TOTAL" "$dataset" &
    running=$((running + 1))
done

# Drain remaining workers.
while [ "$running" -gt 0 ]; do
    wait -n
    running=$((running - 1))
done

# ── Collect results from status markers ────────────────────────────────────
PASSED=()
FAILED=()
SKIPPED=()
for dataset in "${DATASETS[@]}"; do
    status="$STATUS_DIR/${dataset}.status"
    if [ ! -f "$status" ]; then
        FAILED+=("$dataset")
        continue
    fi
    state=$(cut -f1 "$status")
    case "$state" in
        PASS) PASSED+=("$dataset") ;;
        SKIP) SKIPPED+=("$dataset") ;;
        *)    FAILED+=("$dataset") ;;
    esac
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Trial complete"
echo "  Passed:  ${#PASSED[@]}"
echo "  Failed:  ${#FAILED[@]}"
echo "  Skipped: ${#SKIPPED[@]}"
echo "  Results: $PROJECT_ROOT/benchmark_output/runs/$SUITE/"
echo "================================================================"

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed datasets:"
    printf '  - %s\n' "${FAILED[@]}"
fi
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "Skipped datasets (missing script):"
    printf '  - %s\n' "${SKIPPED[@]}"
fi

# Exit non-zero if anything failed so CI/automation notices.
if [ "${#FAILED[@]}" -gt 0 ] || [ "${#SKIPPED[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
