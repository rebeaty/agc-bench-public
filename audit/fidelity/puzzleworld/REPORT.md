# puzzleworld fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (final-answer accuracy only; document stepwise reasoning eval gap; consider stratified sampling)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/puzzleworld_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.puzzleworld_answer_metric.PuzzleWorldAnswerMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2506.06211 (Li et al., PuzzleWorld) - skim
- Repo: https://github.com/MIT-MI/PuzzleWorld - skim (PUZZLE_SYSTEM_PROMPT verbatim port confirmed)
- Dataset: hzli1202/PuzzleWorld (HF, 667 puzzles via `all_metadata.jsonl` + per-puzzle PNG content files)

## Implementation audited
- scenarios/puzzleworld_scenario.py - downloads metadata + images via `hf_hub_download`. `SYSTEM_PROMPT` (lines 59-120) copied verbatim from `src/modeling.py::PUZZLE_SYSTEM_PROMPT` (tips: acrostics, indexing, alpha-numeric codes, anagrams, image manipulations). User prompt (lines 187-197) mirrors `src/reasoner.py::PUZZLE_USER_PROMPT` with title + flavor text + `Answer: <answer>` final-line directive. Images attached as `MediaObject`s in referenced order. 667 puzzles total (easy 140 / medium 355 / hard 172). Optional `PUZZLEWORLD_INSTANCE_LIMIT_HINT` env var for cap.
- metrics/puzzleworld_answer_metric.py - case-insensitive normalized exact match on `solution` (`puzzleworld_final_answer_accuracy`) + parse-rate diagnostic (`puzzleworld_answer_extracted_rate`).
- registry_metrics.yaml: 2 metrics, both `metrics.puzzleworld_answer_metric.PuzzleWorldAnswerMetric` (formula_based).
- registry_inference.yaml: `_use_defaults: true` (T=0.7, max=512, n=1) -- max=512 is far too short for chain-of-thought puzzle solving.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: paper's stepwise reasoning-trace evaluation (GPT-4o-as-judge for path-following + semantic equivalence at each step) is NOT implemented. Only final-answer accuracy is reported. Disclosed in scenario docstring; still a major reported-metric gap.
- [HIGH] D. Generation config: defaults T=0.7, max_new_tokens=512 will truncate the multi-step reasoning the puzzle prompt explicitly requires ("Write out a step-by-step solution... You must end your response with one final line `Answer:`"). Should be max>=4096, T=0.0-0.3.
- [MEDIUM] A. If subsampled (cap-200 via env var), there is no difficulty stratification despite easy/medium/hard labels in extra_data -- could over-sample the dominant medium tier.
- [LOW] B. Prompt: verbatim port of both system + user prompts -- excellent.
- [info] A. 667 puzzles match HF dataset exactly.
- [info] C. No LLM judge wired -- pure formula-based answer match.

## Notes
Prompt fidelity is excellent. The two real risks are (1) the missing stepwise judge metric (paper's secondary result), and (2) generation defaults that truncate chain-of-thought. Patches: bump `max_new_tokens` to 4096+ and drop T to 0.0-0.3 in `registry_inference.yaml`; add a stepwise GPT-4o judge annotator that compares model trace against the human-annotated `reasoning` field; if subsampling, stratify by difficulty.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_caveats
- Now:   Tier 2, high, keep_as_is_with_caveats
- Delta: confirmed
