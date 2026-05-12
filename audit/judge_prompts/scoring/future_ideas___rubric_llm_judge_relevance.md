# future_ideas — _RUBRIC_LLM_JUDGE_RELEVANCE

**Category**: scoring
**Source**: `run_specs/future_ideas_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_RELEVANCE`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the RELEVANCE of the generated future idea to the given domain or topic.
Consider how well the idea addresses the specified area and aligns with the domain's challenges.

Score 1: Idea has no relevance to the specified domain or topic
Score 2: Minimal relevance with major off-topic elements
Score 3: Somewhat relevant but with notable gaps or off-topic elements
Score 4: Mostly relevant and well-aligned with the domain
Score 5: Perfectly relevant and highly focused on the domain's core challenges
```
