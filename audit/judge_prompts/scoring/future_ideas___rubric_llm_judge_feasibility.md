# future_ideas — _RUBRIC_LLM_JUDGE_FEASIBILITY

**Category**: scoring
**Source**: `run_specs/future_ideas_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_FEASIBILITY`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the FEASIBILITY of the generated future idea.
Consider whether the proposed direction is realistic, scientifically grounded, and plausibly investigable.

Score 1: Idea is infeasible or scientifically implausible
Score 2: Idea has major feasibility problems or missing prerequisites
Score 3: Idea is partially feasible but has notable practical or scientific gaps
Score 4: Idea is mostly feasible with reasonable methods or assumptions
Score 5: Idea is highly feasible, well-grounded, and realistic to investigate
```
