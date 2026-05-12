# mops — _PROMPTS['originality_score']

**Category**: scoring
**Source**: `llm_judge/mops_annotator.py`
**Notes**: Module-level string constant `_PROMPTS['originality_score']`. Verbatim from source paper rubric.

```text
Here is a story premise:

{premise}

Now let you give a score from 0 to 100 which represents your level of familiarity with it.

Score 0 indicates that you have seen the exact same premise, while score 100 indicates that you have never seen the same premise at all.

Your score should be based on the assumption that the candidate is at least a complete story premise. Otherwise, you should give a score 0.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:
```
