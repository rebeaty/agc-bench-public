# AGC-Human counterfactual style-flip — POLISH_PROMPT

**Category**: agc_human
**Source**: `scripts/style_flip_counterfactual.py`
**Judge model**: `AGC-Judge`
**Notes**: Counterfactual prompts swap a human response into LLM-style register and vice versa. App L.

```text
You will rewrite a response to sound like a polished AI language model produced it on a creativity-task survey. Critical constraints:

1. PRESERVE EVERY IDEA exactly. Do not add, remove, or substitute any concept.
2. Match the same number of distinct ideas as the original.
3. Raise the register: use precise vocabulary, formal grammar, structured phrasing, light literary or technical flourishes where natural.
4. Keep approximately the same length.
5. Do NOT make it more creative — only change the *style*, not the content.

Output ONLY the rewritten response. No preamble, no explanation, no quote marks.

Original response:
{response}

Rewritten as polished LLM output:
```
