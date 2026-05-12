# liveideabench — _FLUENCY_PROMPT

**Category**: scoring
**Source**: `llm_judge/liveideabench_annotator.py`
**Notes**: Module-level string constant `_FLUENCY_PROMPT`. Verbatim from source paper rubric.

```text
Here are two ideas submitted to "Good Scientific Ideas" Competition, which both relate to "{keyword}":

# The first idea

{idea_a}

# The second idea

{idea_b}


# Question

Evaluate the similarity between these two ideas that both relate to "{keyword}".
Please choose the best answer:

A. Completely different ideas addressing different problems, despite relating to the same keyword.
B. Different ideas but addressing similar problems.
C. Similar ideas addressing similar or identical problems.
D. Academically identical ideas with the same core approach and problem statement.

ONLY ANSWER A/B/C/D, DO NOT EXPLAIN
```
