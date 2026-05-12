#!/usr/bin/env bash
# Serve AGC-Judge locally via vLLM on a single 80GB+ GPU.
#
# This is the BYO-GPU alternative to HF Inference Endpoints. The model is
# downloaded once from HF (~60GB base + ~150MB LoRA adapter) and served
# at http://localhost:8000/v1 with vLLM's continuous batching.
#
# Same OpenAI-compatible /v1/chat/completions surface as the HF endpoint —
# HELM's HFInferenceEndpointClient calls it the same way; only the
# `base_url:` field in prod_env/model_deployments.yaml differs.
#
# Usage:
#   ./eval_scripts/serve_agc_judge_locally.sh                    # foreground; Ctrl-C to stop
#   ./eval_scripts/serve_agc_judge_locally.sh --port 8123        # custom port
#
# Prerequisites:
#   - Single GPU with >= 80GB VRAM (H100, H200, A100-80GB, B200, MI300X)
#   - vllm >= 0.6.4: `pip install "vllm>=0.6.4"`
#   - HF token via `huggingface-cli login` or HF_TOKEN env var (the LoRA repo
#     is public so this is only needed if HF rate-limits anonymous downloads)
#
# After starting, the script prints a one-line command that updates
# update prod_env/model_deployments.yaml. Then in another terminal:
#   ./eval_scripts/run_with_agc_judge.sh openai/gpt-5.5
#
# Quantization (24-48 GB GPUs): pass --quantization int8 or --quantization awq
# when invoking; vLLM will load AGC-Judge in lower precision. Quantized serving
# is not part of the released calibration; the JRT calibration used 16-bit.
set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────
PORT=8000
HOST=0.0.0.0
BASE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
LORA_REPO="agcbench-2026/AGC-Judge"
LORA_NAME="agc-judge"
GPU_UTIL=0.92
# 16k context handles the long-prompt benches (writingbench, rpgbench, etc.).
# At 7k, prompts overflow and the annotator falls back to BACKUP_JUDGE_MODEL,
# which defeats the AGC_JUDGE_OVERRIDE pin and burns external-API quota.
MAX_MODEL_LEN=24576

# ── Parse args ─────────────────────────────────────────────────────────────
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)              PORT="$2"; shift 2 ;;
        --host)              HOST="$2"; shift 2 ;;
        --base-model)        BASE_MODEL="$2"; shift 2 ;;
        --lora-repo)         LORA_REPO="$2"; shift 2 ;;
        --gpu-util)          GPU_UTIL="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0 ;;
        *)  EXTRA_ARGS+=("$1"); shift ;;  # forward unknown args (e.g. --quantization int8) to vllm serve
    esac
done

# ── GPU check ──────────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Need a CUDA-capable GPU to run locally." >&2
    echo "       Use ./eval_scripts/spin_up_hf_endpoint.py for hosted alternative." >&2
    exit 2
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ -z "$VRAM_MB" ]; then
    echo "ERROR: could not query GPU memory via nvidia-smi." >&2
    exit 2
fi
if [ "$VRAM_MB" -lt 75000 ]; then
    echo "WARNING: GPU has ${VRAM_MB} MB VRAM. AGC-Judge in 16-bit needs ~75GB."
    echo "         Loading is likely to OOM. Pass --quantization int8 (or awq)"
    echo "         to attempt 8-bit / 4-bit loading. JRT calibration was done"
    echo "         in 16-bit; lower-precision results may drift."
    echo
fi

# ── vLLM availability ──────────────────────────────────────────────────────
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not installed. Run: pip install \"vllm>=0.6.4\"" >&2
    exit 3
fi

VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo unknown)
echo "vLLM: ${VLLM_VERSION}"

# ── Banner ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  AGC-Judge local server (vLLM)"
echo "  base model:    ${BASE_MODEL}"
echo "  LoRA adapter:  ${LORA_REPO}  (named '${LORA_NAME}' in OpenAI requests)"
echo "  serving on:    http://${HOST}:${PORT}/v1"
echo "  VRAM detected: ${VRAM_MB} MB"
echo "  extra args:    ${EXTRA_ARGS[*]:-(none)}"
echo "================================================================"
echo
echo "After it boots (~2-5 min on first run), update model_deployments.yaml:"
echo
echo "  sed -i 's|base_url: .*|base_url: http://localhost:${PORT}/v1|' prod_env/model_deployments.yaml"
echo
echo "Then in another terminal:"
echo
echo "  ./eval_scripts/run_with_agc_judge.sh <model>"
echo
echo "Ctrl-C to stop the server."
echo

# ── Launch ─────────────────────────────────────────────────────────────────
exec vllm serve "${BASE_MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "${LORA_NAME}=${LORA_REPO}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
