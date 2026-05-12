"""HELM client for serving AGC-Judge via either HF Inference Endpoints or
a local vLLM process. Both expose an OpenAI-compatible /v1/chat/completions
surface, which HELM's stock `OpenAIClient` already speaks via `base_url`.

This thin subclass exists so that:
  - The model_deployments.yaml `client_spec.class_name` points at a clearly
    named class that identifies the AGC-Judge endpoint client rather than the
    generic OpenAI-compatible base client.
  - The api_key default is empty-string-allowed: local vLLM accepts any
    bearer header (including a placeholder), while HF Inference Endpoints
    requires a real HF token. Either way the deployment yaml controls it.

Same mechanical pattern as `helm.clients.stanfordhealthcare_openai_client.StanfordHealthCareOpenAIClient`
in HELM main — see the canonical custom-client example in
src/helm/config/model_deployments.yaml.

Usage in model_deployments.yaml:

    model_deployments:
      - name: agcbench-2026/agc-judge
        model_name: agcbench-2026/AGC-Judge
        tokenizer_name: Qwen/Qwen3-30B-A3B-Instruct-2507
        client_spec:
          class_name: llm_judge.hf_inference_endpoint_client.HFInferenceEndpointClient
          args:
            base_url: http://localhost:8000/v1
            api_key: not-required-for-local-vllm
"""
from __future__ import annotations
from typing import Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig


class HFInferenceEndpointClient(OpenAIClient):
    """OpenAI-compatible client for AGC-Judge endpoints (HF or local vLLM).

    Inherits OpenAIClient verbatim — the only reason to subclass is naming +
    the empty-api_key default for local vLLM serving.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Local vLLM doesn't require auth, but the OpenAI Python SDK still wants
        # something non-None to set the bearer header. Pass a placeholder if the
        # deployment yaml didn't.
        if not api_key:
            api_key = "not-required-for-local-vllm"
        super().__init__(
            cache_config=cache_config,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
