# mops — _PROMPTS['completeness_score']

**Category**: scoring
**Source**: `llm_judge/mops_annotator.py`
**Notes**: Module-level string constant `_PROMPTS['completeness_score']`. Verbatim from source paper rubric.

```text
Here is a story premise:

{premise}

Now let's give you a score from 0 to 100 which represents its completeness level.

Score 0 indicates that it lacks all elements , while score 100 indicates that it has all elements.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:
```
