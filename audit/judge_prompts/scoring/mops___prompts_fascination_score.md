# mops — _PROMPTS['fascination_score']

**Category**: scoring
**Source**: `llm_judge/mops_annotator.py`
**Notes**: Module-level string constant `_PROMPTS['fascination_score']`. Verbatim from source paper rubric.

```text
Here is a story premise:

{premise}

Now let's give you a score from 0 to 100 to assess to its fascination.

Score 0 indicates that this premise is completely confused, while score 100 indicates that you really want to see the story created based on this premise.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:
```
