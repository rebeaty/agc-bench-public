# AGC-Human fairness-aware judge prompt

**Category**: agc_human
**Source**: `scripts/cap_promptD_fairness_detection.py`
**Judge model**: `AGC-Judge (Qwen3-30B-A3B LoRA)`
**Notes**: Discloses LLM self-preference, asks judge to focus on idea over register, labels source. App L.

```text
You will see creative responses written by either human study participants or by AI language models. AI judges sometimes show self-preference bias, rating LLM responses higher than human responses on the same task. Apply the same scale fairly to both sources.

Task description: {p}

Response: {r}

A creative response is novel, high-quality, and surprising. Predict what a panel of human raters would assign on a 10-50 scale, then identify the likely source.

Reply with exactly this format (no other text):
<integer 10-50>|<human or llm>

Example: 32|human
```
