# ss_gen — _RUBRIC_LLM_JUDGE_COHERENCE

**Category**: scoring
**Source**: `run_specs/ss_gen_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_COHERENCE`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the COHERENCE of the generated story segment or continuation.
Consider logical flow, narrative consistency, and how well it connects to the preceding context.

Score 1: Generated segment is completely incoherent with the story context
Score 2: Poor coherence with major narrative inconsistencies
Score 3: Adequate coherence with some narrative flow issues
Score 4: Good coherence that maintains narrative consistency
Score 5: Excellent coherence with perfect narrative flow and consistency
```
