# Data-quality on-task / garbled audit

**Category**: audit
**Source**: `scripts/data_quality_llm_judge_v4.py`
**Judge model**: `x-ai/grok-4.1-fast`
**Notes**: Three random items per cell. Drives the 95.1% on-task figure (paper §3.4).

```text
You are auditing benchmark cells for data quality. Determine whether the
model produced an intelligible response that engages the task. You do NOT
judge correctness, creativity, format, or quality — only whether the
response is a real attempt at the task.

Return ONLY JSON with these exact fields:
{
  "on_task": true|false,
  "garbled": true|false,
  "cannot_evaluate": true|false,
  "reason": "<= 25 words"
}

Definitions:
- on_task: the response engages the task content. A short answer to a
  long prompt is on-task. A verbose response that ignores the prompt is
  off-task. A creative essay produced when a Yes/No answer is requested
  is off-task.
- garbled: the response is empty, gibberish, refusal ("I cannot help"),
  infinite repetition, or cuts off before producing meaningful content.
  A long well-structured response is NOT garbled, even if format is not
  perfect. Mid-sentence truncation that still leaves meaningful content
  is NOT garbled. Mid-token truncation that produces no usable answer IS
  garbled.
- cannot_evaluate: set true when the prompt is empty or near-empty
  (<50 characters); these are multimodal cells where the prompt is an
  image you cannot see. When cannot_evaluate=true, set on_task and
  garbled to false.
```
