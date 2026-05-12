#!/usr/bin/env bash
# Shared helper: runs helm-run with AGC-Bench run_specs registered.
# Source project root onto PYTHONPATH, then invoke HELM via Python so that
# our @run_spec_function decorators fire in the same process.
#
# Usage (from other eval scripts):
#   source "$(dirname "$0")/_helm_run.sh"
#   helm_run "${HELM_ARGS[@]}"

_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
_HELM_PYTHON="${AGC_PYTHON_BIN:-python}"
if [[ -n "${AGC_PYTHON_BIN:-}" ]]; then
    _HELM_BIN_DIR="$(cd "$(dirname "${_HELM_PYTHON}")" && pwd)"
    export PATH="${_HELM_BIN_DIR}${PATH:+:${PATH}}"
fi

helm_preflight() {
    "${_HELM_PYTHON}" - <<'PY'
import importlib
import sys

missing = []
for module in ["helm", "run_specs"]:
    try:
        importlib.import_module(module)
    except Exception as e:
        missing.append((module, repr(e)))

if missing:
    print("ERROR: HELM runtime preflight failed.", file=sys.stderr)
    for module, err in missing:
        print(f"  - {module}: {err}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Install the benchmark package and HELM into the active environment first.", file=sys.stderr)
    print("Suggested commands:", file=sys.stderr)
    print('  pip install -e ".[eval,dev]"', file=sys.stderr)
    sys.exit(1)
PY
}

helm_run() {
    helm_preflight || return $?

    local helm_args=("$@")

    # patch_2026_04_26_threadbump: default num-threads to 16 if not specified
    local has_num_threads=0
    local arg
    for arg in "${helm_args[@]}"; do
        if [[ "${arg}" == "--num-threads" || "${arg}" == "-n" ]]; then
            has_num_threads=1; break
        fi
    done
    if [[ "${has_num_threads}" == "0" ]]; then
        # Prefer AGC_-prefixed knob; fall back to legacy ABC_-prefixed for back-compat.
        helm_args+=("--num-threads" "${AGC_NUM_THREADS:-${ABC_NUM_THREADS:-16}}")
    fi

    # Make HELM read prod_env/{model_deployments,model_metadata,credentials.conf}
    # from the project root, regardless of CWD when this is invoked.
    local has_local_path=0
    local arg
    for arg in "${helm_args[@]}"; do
        if [[ "${arg}" == "--local-path" ]]; then
            has_local_path=1; break
        fi
    done
    if [[ "${has_local_path}" == "0" ]]; then
        helm_args+=("--local-path" "${_PROJECT_ROOT}/prod_env")
    fi
    if [[ "${AGC_DISABLE_HELM_CACHE:-${ABC_DISABLE_HELM_CACHE:-0}}" == "1" ]]; then
        local has_disable_cache=0
        local arg
        for arg in "${helm_args[@]}"; do
            if [[ "${arg}" == "--disable-cache" ]]; then
                has_disable_cache=1
                break
            fi
        done
        if [[ "${has_disable_cache}" == "0" ]]; then
            helm_args+=("--disable-cache")
        fi
    fi

    "${_HELM_PYTHON}" -c "
import sys, run_specs
from helm.benchmark.run import main
sys.exit(main())
" "${helm_args[@]}"
}
