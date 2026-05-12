# conceptual_design — _RUBRIC_CONCEPTUAL_DESIGN

**Category**: scoring
**Source**: `run_specs/conceptual_design_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_CONCEPTUAL_DESIGN`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the full set of generated conceptual design solutions for an engineering problem.
Consider the set as a whole and score the following dimensions on the anchored 0-2 scale.

1. feasibility: 0 = infeasible or not implementable, 1 = partially feasible, 2 = feasible and implementable
2. novelty: 0 = common or repetitive, 1 = somewhat novel, 2 = clearly novel and distinct from the reference pool
3. usefulness: 0 = off-topic or unhelpful, 1 = somewhat useful, 2 = useful and relevant to the prompt

Respond with a JSON object containing only the three integer scores.
```
