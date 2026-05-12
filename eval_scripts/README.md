# `eval_scripts/` - benchmark run entry points

This directory contains shell wrappers for running AGC-Bench through HELM.

| Entry point | Purpose |
|---|---|
| `run_with_agc_judge.sh` | Recommended end-to-end path for scoring a new model on the 67 primary text-only datasets with AGC-Judge for LLM-judge cells. |
| `00_run_all_parallel.sh` | Runs all per-benchmark scripts in parallel. |
| `00_run_all.sh` | Sequential runner, useful for debugging. |
| `<benchmark>.sh` | Runs one benchmark-specific HELM run. |
| `spin_up_hf_endpoint.py` / `tear_down_hf_endpoint.py` | Optional helpers for creating or deleting an AGC-Judge HF Inference Endpoint. |
| `serve_agc_judge_locally.sh` | Optional local-vLLM serving helper for AGC-Judge. |

The underlying HELM run definitions are in `run_specs/`.
