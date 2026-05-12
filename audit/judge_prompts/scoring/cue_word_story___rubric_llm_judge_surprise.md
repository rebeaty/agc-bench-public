# cue_word_story — _RUBRIC_LLM_JUDGE_SURPRISE

**Category**: scoring
**Source**: `run_specs/cue_word_story_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_SURPRISE`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the SURPRISE of the generated short story.
Consider how unexpected the story is, including twists, reversals, and unusual
transitions between ideas.

Score 1: Completely predictable; no unexpected turns
Score 2: Mostly predictable with almost no surprise
Score 3: Some mildly unexpected ideas or transitions
Score 4: Clearly surprising with at least one strong twist or reversal
Score 5: Highly surprising and inventive with memorable unexpected turns
```
