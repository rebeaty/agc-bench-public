# CAP validity gate (shared by AGC-Human release-set + be-creative + reasoning interventions)

**Category**: audit
**Source**: `scripts/score_cap_validity_judge.py`
**Judge model**: `openai/gpt-5.4-mini (single judge); be-creative intervention extends this to a 3-judge consensus`
**Notes**: Per-(entity, task, prompt) validity check. Drives the 96.0% (humans) / 98.8% (LLMs) AGC-Human pass rate (paper §3.7) AND the 99.4% pass rate on the 2,700 be-creative generations (paper §4.6). The be-creative experiment imports PROMPT_TEMPLATE from this module, so the two validity gates share an identical prompt.

```text
You are checking whether a participant gave a valid attempt at a creativity task.

Task: {task_description}
Item: {prompt_label}

Response:
"""
{response}
"""

Is this a valid attempt at the task? A valid response addresses the prompt with a
sensible attempt — even if the answer is short, ordinary, or unimaginative, it
counts as VALID as long as it's on-topic and not gibberish.

INVALID means the response is empty, gibberish, refuses to answer, repeats the
prompt, or is wildly off-topic.

Reply with exactly one word: VALID or INVALID
```
