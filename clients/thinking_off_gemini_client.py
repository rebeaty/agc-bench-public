"""Gemini client that disables internal reasoning ("thinking") for gemini-3*
models when used as an LLM judge.

Why this exists
---------------
Gemini 3 series models (gemini-3-flash-preview, gemini-3-pro-preview, etc.)
default to thinking-on. Like OpenRouter reasoning models, they emit hidden
thoughts that consume the `max_output_tokens` budget BEFORE any visible
output is produced. With a typical judge prompt and `max_tokens=50`, the
budget is spent entirely on thoughts and the API returns:

    finishReason: MAX_TOKENS
    content: {}        # empty

HELM's annotators see empty raw output, fail to parse a verdict (preference
becomes nan), and per-instance stats turn into all-zero metrics. This is the
direct-Google-API parallel of the OpenRouter reasoning bug fixed by
ReasoningOffOpenRouterClient.

The Gemini API supports `generationConfig.thinkingConfig.thinkingBudget=0`
to skip thinking and give the full token budget to the answer:
https://ai.google.dev/gemini-api/docs/thinking

The parent GoogleGenAIClient already honors a `thinking_config` constructor
arg, but we apply the override here defensively in `_convert_request_to_
generate_content_config` so it can't be lost to a YAML wiring mistake. We
only force thinking-off for model names matching `gemini-3*` so non-thinking
deployments (gemini-2.5-flash) keep their natural behavior.
"""

from helm.clients.google_genai_client import GoogleGenAIClient
from google.genai.types import GenerateContentConfig, ThinkingConfig
from helm.common.request import Request


class ThinkingOffGoogleGenAIClient(GoogleGenAIClient):
    """GoogleGenAIClient subclass that forces thinking_budget=0 for gemini-3*."""

    @staticmethod
    def _is_thinking_model(model_name: str) -> bool:
        return model_name is not None and model_name.startswith("gemini-3")

    def _convert_request_to_generate_content_config(self, request: Request) -> GenerateContentConfig:
        config = super()._convert_request_to_generate_content_config(request)
        model_name = self._get_model_name_for_request(request)
        if self._is_thinking_model(model_name):
            config.thinking_config = ThinkingConfig(thinking_budget=0)
        return config
