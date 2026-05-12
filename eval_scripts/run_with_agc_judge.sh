#!/usr/bin/env bash
# End-to-end: evaluate a model on AGC-Bench with AGC-Judge as the LLM-judge.
#
# Two AGC-Judge serving paths, both fully HELM-canonical:
#
#   (a) HF Inference Endpoints (default; bring-your-own HF account, scale-to-zero)
#       Pre-deploy with `python eval_scripts/spin_up_hf_endpoint.py` once,
#       then the URL in prod_env/model_deployments.yaml is reused.
#
#   (b) Local vLLM (--local-vllm flag; requires an 80GB+ GPU)
#       This script spawns `vllm serve` in the background, waits for /health,
#       points HELM at http://localhost:8000/v1, runs the eval, and tears
#       down the server on exit (trap-protected). One command, no HF account.
#
# Either path produces identical HELM stats.json output — the integrator
# (scripts/integrate_new_model.py) doesn't care which one served the judge.
#
# Usage:
#   ./eval_scripts/run_with_agc_judge.sh MODEL [N_INSTANCES] [PARALLELISM]
#   ./eval_scripts/run_with_agc_judge.sh --local-vllm MODEL [N_INSTANCES] [PARALLELISM]
#   LOCAL_VLLM=1 ./eval_scripts/run_with_agc_judge.sh MODEL [N_INSTANCES] [PARALLELISM]
#
# Examples:
#   # HF Inference Endpoints (after running spin_up_hf_endpoint.py once)
#   ./eval_scripts/run_with_agc_judge.sh openai/gpt-5.5
#
#   # Local vLLM on your own GPU (no hosted endpoint)
#   ./eval_scripts/run_with_agc_judge.sh --local-vllm anthropic/claude-opus-4.7 50 8
#
# Prerequisites:
#   - HELM installed: `pip install -e ".[eval,dev]"`
#   - prod_env/credentials.conf with `openrouterApiKey:` and (path-a only)
#     `huggingfaceApiToken:` plus `googleApiKey:` for embedding-routed benches
#   - Path (a): HF endpoint URL already in prod_env/model_deployments.yaml
#   - Path (b): vllm>=0.6.4 + 80GB+ GPU on the same machine
set -euo pipefail

# ── Parse local-vLLM mode ───────────────────────────────────────────────────
LOCAL_VLLM="${LOCAL_VLLM:-0}"
if [[ "${1:-}" == "--local-vllm" ]]; then
    LOCAL_VLLM=1
    shift
fi

MODEL=${1:?"Usage: $0 [--local-vllm] MODEL [N_INSTANCES] [PARALLELISM]"}
N_INSTANCES=${2:--1}
PARALLELISM=${3:-4}

cd "$(dirname "$0")/.."

# Route every LLM-judge call through AGC-Judge via the model identifier registered in
# prod_env/model_deployments.yaml. resolve_judge() in
# llm_judge/_judge_override.py reads this env var with priority over each
# bench's per-bench JUDGE_MODEL_OVERRIDE.
export AGC_JUDGE_OVERRIDE="${AGC_JUDGE_OVERRIDE:-agcbench-2026/agc-judge}"

# Embedding-driven metrics (sdat, conceptual_design, slang_generation,
# mops_diversity, semantic_diversity, etc. — see metrics/embedder_factory.py
# header for the full list) need a per-call embedding model. The published
# cohort used Gemini's `gemini-embedding-001`; some accounts may hit
# quota/billing limits depending on their Google API configuration. Users
# without Google billing can set AGC_EMBEDDING_BACKEND=qwen to swap in a local
# Qwen3-Embedding-0.6B model with no third-party calls after model download.
# The override applies only when not already set, so an explicit
# AGC_EMBEDDING_BACKEND=gemini still wins. The legacy ABC_EMBEDDING_BACKEND
# name is still honored by metrics/embedder_factory.py for back-compat.
export AGC_EMBEDDING_BACKEND="${AGC_EMBEDDING_BACKEND:-${ABC_EMBEDDING_BACKEND:-qwen}}"

# Local vLLM usually needs most of the serving GPU. If the user uses the
# local Qwen embedder, keep it on CPU by default so it doesn't compete for VRAM.
# Power users can override with AGC_QWEN_EMBEDDING_DEVICE=cuda:1, etc.
if [[ "${LOCAL_VLLM:-0}" == "1" ]] \
    && [[ "${AGC_EMBEDDING_BACKEND}" == "qwen" ]] \
    && [[ -z "${AGC_QWEN_EMBEDDING_DEVICE:-}" ]] \
    && [[ -z "${ABC_QWEN_EMBEDDING_DEVICE:-}" ]]; then
    export AGC_QWEN_EMBEDDING_DEVICE=cpu
    echo "  qwen embedding device:  cpu (override with AGC_QWEN_EMBEDDING_DEVICE)"
fi

# BERTScore-based metrics can otherwise auto-select the same CUDA device used
# by local AGC-Judge serving.
if [[ "${LOCAL_VLLM:-0}" == "1" ]] && [[ -z "${AGC_BERT_SCORE_DEVICE:-}" ]]; then
    export AGC_BERT_SCORE_DEVICE=cpu
    echo "  bert-score device:      cpu (override with AGC_BERT_SCORE_DEVICE)"
fi

# Per-bench scripts launch HELM via _helm_run.sh, which respects AGC_PYTHON_BIN.
# Auto-discover an interpreter that can `import helm` so a fresh
# `pip install -e ".[eval,dev]"` works without the user having to know about
# this env var. Order: explicit > current `python` > common conda envs.
discover_helm_python() {
    if [[ -n "${AGC_PYTHON_BIN:-}" ]]; then
        if "${AGC_PYTHON_BIN}" -c "import helm" 2>/dev/null; then
            echo "${AGC_PYTHON_BIN}"; return 0
        fi
        echo "ERROR: AGC_PYTHON_BIN=${AGC_PYTHON_BIN} cannot import helm." >&2
        return 1
    fi
    local cand
    for cand in python python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import helm" 2>/dev/null; then
            command -v "$cand"; return 0
        fi
    done
    # Conda envs in the common location.
    local conda_envs="${CONDA_ENVS_PATH:-$HOME/miniconda3/envs}"
    if [ -d "$conda_envs" ]; then
        local env_py
        for env_py in "$conda_envs"/*/bin/python; do
            [ -x "$env_py" ] || continue
            if "$env_py" -c "import helm" 2>/dev/null; then
                echo "$env_py"; return 0
            fi
        done
    fi
    return 1
}

if ! AGC_PYTHON_BIN_AUTO=$(discover_helm_python); then
    echo "ERROR: no Python interpreter on PATH (or under \$CONDA_ENVS_PATH) can import HELM." >&2
    echo "       Install with: pip install -e \".[eval,dev]\"" >&2
    echo "       Or export AGC_PYTHON_BIN=/path/to/python with HELM installed." >&2
    exit 5
fi
export AGC_PYTHON_BIN="$AGC_PYTHON_BIN_AUTO"
echo "  python (HELM-capable):  $AGC_PYTHON_BIN"

# ── Path (b): spawn local vLLM server, point HELM at localhost ─────────────
VLLM_PID=""
LOCAL_PORT=8000
DEPLOYMENTS_BACKUP=""
cleanup_local_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        echo
        echo "--- Tearing down local vLLM server (pid=$VLLM_PID) ---"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    if [[ -n "$DEPLOYMENTS_BACKUP" ]] && [[ -f "$DEPLOYMENTS_BACKUP" ]]; then
        mv "$DEPLOYMENTS_BACKUP" prod_env/model_deployments.yaml
    fi
}
trap cleanup_local_vllm EXIT INT TERM

if [[ "$LOCAL_VLLM" == "1" ]]; then
    echo "--- Phase 0: spawning local vLLM server ---"
    if ! python3 -c "import vllm" 2>/dev/null; then
        echo "ERROR: vllm not installed. Run: pip install \"vllm>=0.6.4\"" >&2
        exit 2
    fi

    # Back up the current deployments yaml; we'll restore on exit
    DEPLOYMENTS_BACKUP=$(mktemp)
    cp prod_env/model_deployments.yaml "$DEPLOYMENTS_BACKUP" 2>/dev/null || true

    # Spawn vLLM serve in the background
    bash eval_scripts/serve_agc_judge_locally.sh --port "$LOCAL_PORT" \
        > /tmp/agcjudge_vllm.log 2>&1 &
    VLLM_PID=$!
    echo "  vLLM pid: $VLLM_PID  log: /tmp/agcjudge_vllm.log"

    # Wait for /health to return 200 (or fail fast if vLLM exits early)
    echo -n "  waiting for vLLM to serve"
    for i in $(seq 1 90); do  # up to 15 min
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo
            echo "ERROR: vLLM exited during startup. Tail of log:" >&2
            tail -30 /tmp/agcjudge_vllm.log >&2 || true
            exit 3
        fi
        if curl -sf "http://localhost:${LOCAL_PORT}/health" >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 10
    done

    # Patch prod_env/model_deployments.yaml to point at localhost
    cat > prod_env/model_deployments.yaml <<EOF
# AGC-Judge served by the local vLLM process (auto-generated by
# eval_scripts/run_with_agc_judge.sh --local-vllm). Restored to the previous
# value on script exit.

model_deployments:
  - name: agcbench-2026/agc-judge
    model_name: agcbench-2026/AGC-Judge
    tokenizer_name: Qwen/Qwen3-30B-A3B-Instruct-2507
    max_sequence_length: 8192
    client_spec:
      class_name: llm_judge.hf_inference_endpoint_client.HFInferenceEndpointClient
      args:
        base_url: http://localhost:${LOCAL_PORT}/v1
        openai_model_name: agc-judge
EOF
    # Local vLLM doesn't require auth; HFInferenceEndpointClient's bearer
    # is sent but ignored.
    if [[ ! -f prod_env/credentials.conf ]]; then
        echo "huggingfaceApiToken: not-required-for-local-vllm" > prod_env/credentials.conf
    fi
fi

# ── Pre-flight checks ──────────────────────────────────────────────────────
if [ ! -f "prod_env/model_deployments.yaml" ]; then
    echo "ERROR: prod_env/model_deployments.yaml is missing." >&2
    echo "       Either run \`python eval_scripts/spin_up_hf_endpoint.py\`" >&2
    echo "       (HF Inference Endpoints) or pass --local-vllm to this script." >&2
    exit 2
fi

AGC_JUDGE_BASE_URL=$("$AGC_PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
import yaml

path = Path("prod_env/model_deployments.yaml")
data = yaml.safe_load(path.read_text()) or {}
for deployment in data.get("model_deployments", []):
    if deployment.get("name") == "agcbench-2026/agc-judge":
        args = ((deployment.get("client_spec") or {}).get("args") or {})
        print(args.get("base_url", ""))
        sys.exit(0)
sys.exit(1)
PY
) || {
    echo "ERROR: prod_env/model_deployments.yaml has no agcbench-2026/agc-judge deployment." >&2
    echo "       Run \`python eval_scripts/spin_up_hf_endpoint.py\` or pass --local-vllm." >&2
    exit 2
}

JUDGE_PATH="HF/OpenAI-compatible endpoint"
if [[ "$AGC_JUDGE_BASE_URL" == http://localhost:* || "$AGC_JUDGE_BASE_URL" == http://127.0.0.1:* ]]; then
    LOCAL_HEALTH_URL="${AGC_JUDGE_BASE_URL%/v1}/health"
    LOCAL_MODELS_URL="${AGC_JUDGE_BASE_URL%/}/models"
    if curl -sf "$LOCAL_HEALTH_URL" >/dev/null 2>&1 || curl -sf "$LOCAL_MODELS_URL" >/dev/null 2>&1; then
        JUDGE_PATH="existing local vLLM (${AGC_JUDGE_BASE_URL})"
    else
        echo "ERROR: AGC-Judge deployment points at ${AGC_JUDGE_BASE_URL}, but no local server is reachable." >&2
        echo "       Start it with --local-vllm, or run eval_scripts/spin_up_hf_endpoint.py to patch" >&2
        echo "       prod_env/model_deployments.yaml to a hosted HF endpoint." >&2
        exit 2
    fi
fi

SAFE_MODEL=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUT_DIR="analysis/scored/${SAFE_MODEL}"
mkdir -p "$OUT_DIR"

[[ "$LOCAL_VLLM" == "1" ]] && JUDGE_PATH="local vLLM (port ${LOCAL_PORT})"

echo "================================================================"
echo "AGC-Bench end-to-end with AGC-Judge"
echo "  model:                 ${MODEL}"
echo "  judge dispatch:        ${JUDGE_PATH}"
echo "  AGC_JUDGE_OVERRIDE:    ${AGC_JUDGE_OVERRIDE}"
echo "  n_instances:           ${N_INSTANCES} (-1 = all)"
echo "  parallelism:           ${PARALLELISM}"
echo "  out dir:               ${OUT_DIR}"
echo "================================================================"

# ── Phase 1: HELM end-to-end (inference + AGC-Judge + metrics) ─────────────
# Phase 1 returns non-zero if ANY per-bench script failed (per-bench failures
# are expected on partial runs / quota hiccups). Don't let that abort Phase 2 —
# the integrator handles partial cell coverage and reports a missing.csv.
echo
echo "--- Phase 1: HELM end-to-end ---"
PHASE1_RC=0
./eval_scripts/00_run_all_parallel.sh "$MODEL" "$N_INSTANCES" "$PARALLELISM" || PHASE1_RC=$?
if [[ "$PHASE1_RC" -ne 0 ]]; then
    echo "  Phase 1 reported per-bench failures (rc=$PHASE1_RC). Continuing to Phase 2;"
    echo "  per-dataset coverage will be reported in ${OUT_DIR}/missing.csv."
fi

# ── Phase 2: Release-set-comparable z + leaderboard slot ───────────────────
echo
echo "--- Phase 2: Release-set integration ---"
python3 scripts/integrate_new_model.py \
    --new-model "$MODEL" \
    --suite first_full_trial \
    --out "$OUT_DIR"

echo
echo "================================================================"
echo "Done. Outputs in ${OUT_DIR}/:"
echo "  per_dataset_z.csv           — per-bench raw + z + canonical metric"
echo "  per_domain.csv              — per-domain mean-z"
echo "  leaderboard_line.json       — composite + nearest-above / -below"
echo "  missing.csv                 — datasets HELM did not produce stats.json for"
echo "================================================================"
