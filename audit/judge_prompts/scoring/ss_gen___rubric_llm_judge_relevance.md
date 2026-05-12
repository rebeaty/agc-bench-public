# ss_gen — _RUBRIC_LLM_JUDGE_RELEVANCE

**Category**: scoring
**Source**: `run_specs/ss_gen_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_RELEVANCE`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the RELEVANCE of the generated Social Story to the requested title and social situation.
Consider whether the story stays on topic and addresses the intended intervention goal.

Score 1: Story is off-topic or fails to address the requested title
Score 2: Weak relevance with major gaps in topic alignment
Score 3: Moderately relevant but with some drift or missing focus
Score 4: Clearly relevant and well-aligned with the requested topic
Score 5: Highly relevant, focused, and directly responsive to the requested title and purpose
```
