# data_narrative — _RUBRIC_DATA_NARRATIVE_FACTUAL_CORRECTNESS

**Category**: scoring
**Source**: `run_specs/data_narrative_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_DATA_NARRATIVE_FACTUAL_CORRECTNESS`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the FACTUAL CORRECTNESS of the generated data narrative.
Consider whether numeric statements, comparisons, trends, and other claims are
supported by the provided table and avoid hallucinated or incorrect details.

Score 1: Pervasively incorrect or hallucinatory
Score 2: Multiple factual mistakes or unsupported claims
Score 3: Mostly correct but with at least one meaningful factual issue
Score 4: Factually solid with only minor imprecision
Score 5: Factually accurate and fully grounded in the table
```
