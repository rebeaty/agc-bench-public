# arastories — _make_paper_aligned_rubric

**Category**: scoring
**Source**: `run_specs/arastories_run_specs.py`
**Notes**: Module-level string constant `_make_paper_aligned_rubric`. Verbatim from source paper rubric. Rubric-building function `_make_paper_aligned_rubric`; final prompt is built per call with metric/dimension args.

```text
def _make_paper_aligned_rubric(metric_name: str, definition: str) -> str:
    criterion_label = metric_name.replace("_", " ").title()
    return f"""\
You are an expert in Arabic language, its dialects, and storytelling. I would like your help in evaluating a story written by a student based on a set of instructions.

Evaluate only this criterion on a 1-5 scale:
{criterion_label}: {definition}

Give the score directly without explanations or additions.
"""
```
