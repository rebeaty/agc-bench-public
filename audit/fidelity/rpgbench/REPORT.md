# rpgbench fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** patch_with_validity_BFS_and_verbatim_judge_prompt (the current `json_validity` is JSON-parse only; paper's "validity" includes BFS-reachability of game scenes/events from the start state. Judge prompt is a paraphrase, not verbatim from `game_interestingness_prompt.txt`.)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/rpgbench_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.json_validity_metric.JsonValidityMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **AnnotatorSpec(s):** `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2502.00595 ("RPGBENCH," ICML 2025) — abstract via arXiv. Confirms two tasks (Game Creation + Game Simulation), objective game-mechanic verification + LLM-as-judge subjective scoring. (note: "skim")
- Repo: https://github.com/boson-ai/rpgbench-public — `data/game_creation/characters.jsonl` and `rpgbench/static/game_creation_prompt.txt`, `game_interestingness_prompt.txt`, `game_json_schema.json` referenced by scenario header. (note: "skim")

## Implementation audited
- scenarios/rpgbench_scenario.py — fetches `data/game_creation/characters.jsonl` (100 Wikipedia bios). Builds prompt by concatenating "Here is a character description: {name}: {description}", a verbatim condensed JSON schema (lines 60-119), and verbatim guidelines from `game_creation_prompt.txt` (lines 122-135). 100 instances, no references (open-ended). **Game Simulation task is correctly excluded** as incompatible with HELM single-turn (acknowledged in scenario header lines 36-38).
- metrics/rpgbench_metric.py — does not exist (`has_metric_file: false`); registry uses shared `metrics/json_validity_metric.py` for structural check.
- registry_metrics.yaml (lines 2294-2317): two metrics — `json_validity` (formula_based, `JsonValidityMetric` — JSON-parse only) and `interestingness` (llm_judge, GPT-4o, T=0, 256 tokens). The judge prompt is filled in (1-5 scale, JSON output dict) but is a **paraphrase** of the upstream `game_interestingness_prompt.txt` rather than verbatim.
- registry_inference.yaml (lines 574-576): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: The paper's "validity" is **structural correctness AND game-mechanic reachability** — checking whether all events and scenes are reachable via BFS from the initial state under variable-update rules. We currently compute only `JsonValidityMetric` (does the output parse as JSON?), missing the entire game-graph BFS validity check. Major undercount on objective evaluation.
- [MEDIUM] B. Prompt fidelity (judge): The interestingness judge prompt is a paraphrase. Should be replaced with the verbatim text from `rpgbench/static/game_interestingness_prompt.txt` for paper-comparability of the subjective score.
- [MEDIUM] A. Task coverage: Game Simulation is excluded (defensible HELM constraint), so we report only the GC half of the paper's primary metrics. Properly disclosed in scenario header.
- [LOW] B. Prompt fidelity (creation): Game-creation prompt body, schema, and guidelines are quoted verbatim from upstream files.
- [info] A. Dataset source: 100 Wikipedia bios; matches paper.
- [info] D. Generation config: defaults; for structured JSON output, low T would be advisable.

## Notes
The scenario side is solid — verbatim creation prompt, schema, and 100-character bio set, plus a sensible decision to skip Game Simulation. The metric side has two real gaps: (1) need a proper `RpgBenchValidityMetric` that does BFS over scenes+events and confirms all are reachable from the initial state under the variable-update model (port from upstream `validate_game.py` if it exists); (2) interestingness judge prompt should be replaced with the verbatim text from `game_interestingness_prompt.txt`. Once those are fixed, this is a clean Tier 2.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium-high, patch_with_validity_BFS_and_verbatim_judge_prompt
- Now:   Tier 2, medium-high, patch_with_validity_BFS_and_verbatim_judge_prompt
- Delta: confirmed
