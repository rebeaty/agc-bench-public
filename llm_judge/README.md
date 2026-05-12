# `llm_judge/` - LLM-judge annotators and AGC-Judge client

This directory contains benchmark-specific annotators and metrics for
LLM-as-judge scoring, plus the OpenAI-compatible client used to route release
evaluation through AGC-Judge.

Important entry points:

| File | Purpose |
|---|---|
| `generic_llm_judge_annotator.py` | Shared annotator for rubric-driven judge calls. |
| `generic_llm_judge_metric.py` | Shared metric wrapper for generic LLM-judge outputs. |
| `_judge_override.py` | Runtime override used by `AGC_JUDGE_OVERRIDE` for release-set-comparable judging. |
| `hf_inference_endpoint_client.py` | HELM client for AGC-Judge endpoints served by HF Inference Endpoints or local vLLM. |

Verbatim judge prompts are collected in `audit/judge_prompts/`.
