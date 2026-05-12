"""OpenRouter client that disables internal reasoning per OpenRouter's
unified `reasoning.enabled=false` parameter.

Why this exists
---------------
Reasoning models (z-ai/glm-5.1, deepseek/deepseek-r1-*, moonshotai/kimi-k2.*,
qwen/qwen3.5-*, openai/gpt-oss-*, etc.) emit internal `<think>` tokens that
consume the `max_tokens` budget BEFORE producing visible output. With the
default per-task max_tokens (often 8 for MCQ tasks), the model spends every
token on reasoning and returns text="". OpenRouter caches that empty as a
successful response, and HELM computes meaningless zero metrics on it.

OpenRouter exposes a unified `reasoning` parameter that, with `enabled=false`,
skips reasoning entirely and gives the full token budget to the answer:
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

The fix is just to inject `extra_body={"reasoning": {"enabled": False}}` into
the raw request dict. Bonus: that field is part of the cache key, so previously
cached empty responses are naturally bypassed without manual invalidation.
"""

from helm.clients.openrouter_client import OpenRouterClient


class ReasoningOffOpenRouterClient(OpenRouterClient):
    def _make_chat_raw_request(self, request):
        raw_request = super()._make_chat_raw_request(request)
        existing = raw_request.get("extra_body") or {}
        existing["reasoning"] = {"enabled": False}
        raw_request["extra_body"] = existing
        return raw_request
