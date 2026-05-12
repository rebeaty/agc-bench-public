# `clients/` - custom HELM clients

These clients handle provider-specific behavior needed by AGC-Bench runs while
keeping HELM run specs stable.

| File | Purpose |
|---|---|
| `reasoning_off_openrouter_client.py` | OpenRouter-compatible client that disables reasoning where required for release-set comparability. |
| `thinking_off_gemini_client.py` | Gemini client variant for runs where thinking/reasoning should be disabled. |
| `gemini_embedding_client.py` | Gemini embedding backend used by embedding-driven metrics when configured. |

Model routing and deployment names are configured in `prod_env/`.
