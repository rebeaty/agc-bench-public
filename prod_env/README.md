# `prod_env/` - deployment configuration templates

This directory contains configuration templates for running AGC-Bench through
HELM. It does not contain credentials.

| File | Purpose |
|---|---|
| `credentials.conf.template` | Template showing the credential names expected by HELM and optional endpoint helpers. |
| `model_deployments.yaml` | Model routing definitions for evaluated models, source-paper judge defaults, AGC-Judge endpoint routing, and embedding backends. |

For a new-model run, start with `eval_scripts/run_with_agc_judge.sh`.
