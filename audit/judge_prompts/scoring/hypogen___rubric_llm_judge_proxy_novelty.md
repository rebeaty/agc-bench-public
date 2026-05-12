# hypogen — _RUBRIC_LLM_JUDGE_PROXY_NOVELTY

**Category**: scoring
**Source**: `run_specs/hypogen_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_PROXY_NOVELTY`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the NOVELTY of the generated hypothesis on the local HypoGen bit-flip proxy task.
Score only whether the candidate proposes an original and interesting "flip" relative to the
provided abstract and conventional limitation ("bit"). Do not treat this as a score for the
full AI-Researcher ideation benchmark from the paper.

Score 1: Hypothesis shows no novel thinking; follows the conventional assumption
Score 2: Minimal novelty; makes only slight modifications to the assumption
Score 3: Somewhat novel hypothesis with partial inversion of the assumption
Score 4: Notably novel hypothesis clearly inverting the assumption in an interesting way
Score 5: Highly novel hypothesis presenting a completely original and insightful inversion
```
