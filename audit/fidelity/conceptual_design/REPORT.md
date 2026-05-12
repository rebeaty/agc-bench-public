# conceptual_design fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium
**Recommendation:** patch_with_metric_registry_alignment

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/conceptual_design_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.conceptual_design_metric.ConceptualDesignMetric`
> - **ScenarioSpec args:** `prompt_variant=prompt_variant`
> - **AnnotatorSpec(s):** `llm_judge.conceptual_design_annotator.ConceptualDesignAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2306.01779 (Ma et al. 2023) — skim
- Repo: https://github.com/kevinma1515/gpt_idetc — skim (zero_shot.py + amazonTurkDesPrompt*.csv referenced)

## Implementation audited
- scenarios/conceptual_design_scenario.py — Clones gpt_idetc, loads 12 problems (peanut/idx 11 excluded), four prompt variants (base/novel/diverse/unique). 100-AMT-solution references attached. Lines 38-77.
- metrics/conceptual_design_metric.py — Set-level SBERT proxies: nearest_reference_similarity, within_set_diversity, solution_count, plus consumer of LLM-judge annotations (feasibility/novelty/usefulness/parse_rate).
- registry_metrics.yaml: rouge_l (formula) + llm_judge_quality (judge=openai/gpt-4, T=0, 256 tokens, prompt=null)
- registry_inference.yaml: `_use_defaults: true` → T=0.7, 512 tokens, n=1

## Deviations found
- [HIGH] C. Registry metric mismatch: registry_metrics.yaml declares `rouge_l` (irrelevant for set-level ideation against 100 human references) and `llm_judge_quality` with `judge_prompt: null` (not auditable). Neither maps to what `ConceptualDesignMetric` actually emits (nearest_reference_similarity, within_set_diversity, etc.).
- [MEDIUM] C. Diversity proxy mismatch: paper's signature diversity metric is SBERT-embedding *convex-hull hypervolume*; we compute mean pairwise cosine distance. Reasonable proxy but not the paper-reported number.
- [MEDIUM] C. Judge swap: GPT-4 stand-in for CAT 3-point human raters of novelty/feasibility/usefulness (canonical substitution). max_new_tokens=256 may be insufficient when scoring 100-solution sets.
- [LOW] B. Prompt fidelity: All four zero-shot variants ("Generate 100 [variant] design solutions for {problem}") reproduced verbatim from `prompt_engineering/zero_shot.py`.
- [LOW] D. Generation config: `_use_defaults` → T=0.7; paper uses T=1.0 for ideation. Set explicitly.
- [info] A. Instance count: 12 problems × 4 variants = 48 instances, matching the paper's evaluable set (peanut excluded — no human refs).

## Notes
The metric implementation file is well-aligned with the paper's spirit (SBERT, set-level), but the registry exposes the wrong metric names. Lowest-effort patch: (1) replace `rouge_l` in registry_metrics.yaml with the seven stat names emitted by `ConceptualDesignMetric.evaluate_instances` (lines 128-140), (2) document the judge prompt and lift max_new_tokens, (3) add convex-hull hypervolume as a stretch goal to claim parity with paper's signature metric, (4) pin T=1.0 in inference registry per paper.

## Compared to prior audit (2026-04-25)
- Prior: Tier ?, unknown, no audit conducted
- Now:   Tier 2, medium, patch_with_metric_registry_alignment
- Delta: newly classified
