# poetmt — _RUBRIC_LLM_JUDGE_BEAUTY_OF_FORM

**Category**: scoring
**Source**: `run_specs/poetmt_run_specs.py`
**Notes**: Module-level string constant `_RUBRIC_LLM_JUDGE_BEAUTY_OF_FORM`. Verbatim from source paper rubric. Defined in the run_spec module.

```text
Evaluate the translation of the given Chinese classical poem into English.
Focus on whether the translation maintains consistency with the source poem's structure, including the alignment of line numbers and balanced phrasing.

1 point: Poor translation, disregards the poem's structure, and fails to convey its aesthetic qualities.
2 point: Some attempt to maintain structure but lack alignment and aesthetic consistency.
3 point: Basic structural elements are maintained but with noticeable imperfections in alignment and phrasing.
4 point: Good translation, with most structural elements preserved and minor issues in phrasing and alignment.
5 point: Excellent translation, accurately preserving the structure, alignment, and aesthetic qualities of the original poem.
```
