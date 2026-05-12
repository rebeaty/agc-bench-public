# thenextchapter — _build_rubric

**Category**: scoring
**Source**: `run_specs/thenextchapter_run_specs.py`
**Notes**: Module-level string constant `_build_rubric`. Verbatim from source paper rubric. Rubric-building function `_build_rubric`; final prompt is built per call with metric/dimension args.

```text
def _build_rubric(label: str, definition: str) -> str:
    return f"""\
Evaluate the {label.upper()} of the generated story continuation for The Next Chapter benchmark.
Consider only this single dimension.

{label.title()}: {definition}

Score 1: Very poor
Score 2: Poor
Score 3: Adequate
Score 4: Good
Score 5: Excellent
"""
```
