# pron_vs_prompt fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** keep_as_is_with_judge_substitution_caveat (reconcile registry vs implementation: registry advertises 3 judge metrics with null prompts; metric file expects 14+ rubric keys from a `pron_vs_prompt_literary_judge` annotator that needs registration)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/pron_vs_prompt_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.pron_vs_prompt_metric.PronVsPromptMetric`
> - **AnnotatorSpec(s):** `llm_judge.pron_vs_prompt_annotator.PronVsPromptAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2407.01119 ("Pron vs Prompt," EMNLP 2024) — abstract via arXiv: 30 titles × Pron + 30 titles × GPT-4, 5,400 manual assessments by critics. Primary = "LLMs are still far from challenging a top human creative writer." (note: "skim")
- Repo: https://github.com/grmarco/pron-vs-prompt — `data/synopses_texts.csv` referenced. (note: "skim")

## Implementation audited
- scenarios/pron_vs_prompt_scenario.py — fetches `data/synopses_texts.csv`. Filters by `language` (en|es) and `title_origin` (all|patricio|machine). Builds verbatim system + user prompts ("We are going to do an experiment in which we are going to compare your creative writing skills with those of a prestigious novelist, Patricio Pron... synopsis of about 600 words..."). Empty references (open-ended). 30+30=60 instances when title_origin="all".
- metrics/pron_vs_prompt_metric.py — `PronVsPromptMetric` reads `request_state.annotations["pron_vs_prompt_literary_judge"]` and emits 6 group means (parse_rate, attractiveness, originality, relevance, creativity, criticism), aggregating from 14 rubric sub-keys (title_attractiveness, style_attractiveness, theme_attractiveness, title_originality, style_originality, plot_originality, relevance, title_creativity, synopsis_creativity, anthology, readers_opinion, critics_opinion, own_voice, parse_rate).
- registry_metrics.yaml (lines 2122-2144): three llm_judge metrics — `llm_judge_attractiveness`, `llm_judge_originality`, `llm_judge_creativity` — all `judge_model_name: openai/gpt-4`, `judge_prompt: null`, T=0.0, 256 tokens. **Does NOT route to `PronVsPromptMetric` and the rubric prompt is unfilled.**
- registry_inference.yaml (lines 524-526): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity (registry/code mismatch): The implemented `PronVsPromptMetric` plus the `pron_vs_prompt_literary_judge` annotator are not registered. The registered judge metrics have `judge_prompt: null` and do not route through the metric file's group-mean aggregator. As-is, the bench either emits empty judge scores or the implemented metric is dead code. Fix: (a) register the `pron_vs_prompt_literary_judge` annotator with verbatim Boden-inspired rubric prompt covering all 14 sub-dimensions; (b) replace the three `llm_judge_*` registry entries with formula_based pointers to `metrics.pron_vs_prompt_metric.PronVsPromptMetric`; (c) document GPT-4 vs Gemini judge swap if applicable.
- [MEDIUM] C. Metric fidelity: Even after wiring, paper's primary result is **expert human pairwise judgments** (5,400 assessments by literary critics). LLM-judge proxy is the only portable option but not paper-primary.
- [LOW] B. Prompt fidelity: System+user prompt verbatim from the spirit of the paper's setup. Bilingual support (en/es) appropriate.
- [info] A. Dataset source: 60 synopsis titles loaded; matches the 30-Pron + 30-machine corpus.
- [info] D. Generation config: defaults; for ~600-word literary writing, a moderate temperature would be appropriate but not specified.

## Notes
The metric file is well-thought-out — 14 named rubric dimensions aggregated into 6 group means matching the paper's evaluative axes. The blocker is purely registration: the annotator and metric class are orphaned in the registry, which instead lists three null-prompt judge metrics that don't connect to anything. Fix is mechanical but required before any results can be generated. Once wired, this is a clean Tier 2 keep_as_is_with_judge_substitution_caveat.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_judge_substitution_caveat (reconcile registry/implementation)
- Now:   Tier 2, medium-high, keep_as_is_with_judge_substitution_caveat
- Delta: confirmed (still requires the implementation/registry reconciliation flagged previously)
