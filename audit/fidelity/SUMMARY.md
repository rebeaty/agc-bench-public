# AGC-Bench v1 fidelity audit — summary

78 onboarded benchmarks audited under the four-axis paper-vs-implementation protocol described in [AUDIT_METHODOLOGY.md](./AUDIT_METHODOLOGY.md). Each `<bench>/REPORT.md` carries a tier, confidence, and a list of severity-tagged deviations (HIGH / MEDIUM / LOW). The italicized text under each entry is the per-bench audit recommendation. For benchmarks whose body discusses registry-only metric layouts that have since been wired into the run-spec, an "Implementation note" at the top names the live MetricSpec / AnnotatorSpec / ScenarioSpec wiring (37 of 78 reports carry such a note); for benchmarks without one, the report body and the corresponding `run_specs/<bench>_run_specs.py` are the canonical reference.

## Tier counts

| Tier | Count | Description |
|---|---|---|
| **1** | 19 | Faithful within HELM tolerance — only LOW deviations on the four axes (instance source, prompt fidelity, metric/scoring fidelity, generation config). |
| **2** | 50 | Notable deviation, spirit preserved — may include HIGH deviations on a single axis (commonly metric/scoring fidelity, where a paper-canonical metric was substituted with a defensible analog). The implemented evaluation produces a coherent signal aligned with the source paper. |
| **3** | 9 | Proxy / adapted scoring — the implemented metric differs in substantive construct from the paper's primary metric. v1 retains Tier-3 benchmarks in the primary release set with the deviation documented per-bench; the [Tier-3 sensitivity check in release_data/SCORING_NOTES.md](../../release_data/SCORING_NOTES.md) reports the delta on primary numbers when the eight included Tier-3 cells are dropped (intelligence ρ shifts ≤ 0.01; c-factor magnitude shrinks but stays unidimensional). |
| **?** | 0 | Insufficient evidence to classify; needs manual review. |

## Tier 1 — keep as-is

- **[analobench](./analobench/REPORT.md)** (High) — *Keep as-is. Optionally add T2/T3/Sx variants; confirm `T1S1-Subset` matches the paper's small-bank (4-option) condition.*
- **[arn](./arn/REPORT.md)** (medium-high) — *keep_as_is (optional patch: register arn_metric.py in registry_metrics.yaml)*
- **[creation_mmbench](./creation_mmbench/REPORT.md)** (Medium-High. Scenario and metric names align with paper; main gap is the dual-judge annotator code/prompt is not visible here, so VFS/Reward computation cannot be verified end-to-end from this repo alone) — *Populate `judge_prompt` in `registry_metrics.yaml` and ensure the `creation_mmbench_judge` annotator (with position-swap…*
- **[grapheval_ai_researcher](./grapheval_ai_researcher/REPORT.md)** (high) — *keep_as_is (optional: expand evaluation set or add small-N caveat)*
- **[grapheval_iclr](./grapheval_iclr/REPORT.md)** (high) — *keep_as_is (minor optional polish)*
- **[historical_analogy](./historical_analogy/REPORT.md)** (high) — *keep_as_is (optional: add general-analogy split + MDS judge)*
- **[infochartqa](./infochartqa/REPORT.md)** (high) — *keep_as_is (optional: pin HF dataset revision for reproducibility)*
- **[irfl](./irfl/REPORT.md)** (high) — *keep_as_is (optional: document generative-track omission)*
- **[mops](./mops/REPORT.md)** (high) — *keep_as_is*
- **[ocw](./ocw/REPORT.md)** (high) — *keep_as_is (verify 556 vs 618 eval pool count)*
- **[permpst](./permpst/REPORT.md)** (high) — *keep_as_is*
- **[rebus_puzzle](./rebus_puzzle/REPORT.md)** (high) — *keep_as_is (minor: judge model swap is documented)*
- **[riddlesense](./riddlesense/REPORT.md)** (high) — *keep_as_is (optional: log accuracy explicitly alongside F1)*
- **[scar](./scar/REPORT.md)** (high) — *keep_as_is (optional: add multi-template sweep)*
- **[simile_generation](./simile_generation/REPORT.md)** (high) — *keep_as_is (Novelty + human eval intentionally out of scope)*
- **[speak_to_structure](./speak_to_structure/REPORT.md)** (high) — *keep_as_is (add MolCustom_BasicProp subtask for full 10/10 coverage)*
- **[ss_gen](./ss_gen/REPORT.md)** (high) — *keep_as_is (human-rater criteria appropriately omitted)*
- **[tinyfabulist](./tinyfabulist/REPORT.md)** (high) — *keep_as_is*
- **[writingbench](./writingbench/REPORT.md)** (high) — *keep_as_is (reconcile judge model between registry and run_spec)*

## Tier 2 — keep with caveat / patch

- **[arastories](./arastories/REPORT.md)** (medium-high) — *patch_with_judge_rubric (populate `judge_prompt` for each of the 5 LLM-judge dimensions; metric .py file optional)*
- **[arena_hard_creative](./arena_hard_creative/REPORT.md)** (high) — *patch_with_judge_metric_module (implement `metrics/arena_hard_creative_metric.py` + populate `judge_prompt`; consider GP…*
- **[artinsight](./artinsight/REPORT.md)** (medium-high) — *patch_with_judge_rubric (populate `judge_prompt` for the six artinsight_* metrics from `description_scorer.py`; optional…*
- **[banner_request_400](./banner_request_400/REPORT.md)** (medium) — *keep_as_is_with_caveat (surface judge prompts in registry; document the local-renderer substitution)*
- **[brainteaser](./brainteaser/REPORT.md)** (high) — *patch_with_registry_wiring (point `registry_metrics.yaml` at `metrics.brainteaser_metric.BrainteaserMetric` and enumerat…*
- **[c3_crosstalk](./c3_crosstalk/REPORT.md)** (high) — *keep_as_is (document deviations: prompt is a synthesized instruction, not a paper template; human-eval dims absent)*
- **[chinese_homophonic_puns](./chinese_homophonic_puns/REPORT.md)** (high) — *keep_as_is_with_caveat (document prompt simplification — paper used batched line-numbered prompt; we use single-instance…*
- **[conceptual_design](./conceptual_design/REPORT.md)** (medium) — *patch_with_metric_registry_alignment*
- **[cue_word_story](./cue_word_story/REPORT.md)** (medium) — *patch_with_sentence_embeddings_and_judge_prompt*
- **[data_narrative](./data_narrative/REPORT.md)** (medium-high) — *keep_as_is_with_judge_prompts (fix null judge prompts before scoring)*
- **[eqbench_creative_writing_v3](./eqbench_creative_writing_v3/REPORT.md)** (high) — *keep_as_is_with_registry_cleanup (remove unimplemented `elo_rating` from registry; document `min_p=0.1` gap)*
- **[esp_dataset](./esp_dataset/REPORT.md)** (medium-high) — *keep_as_is (optional patch: replace prompt with paper's verbatim `"{style}:"` prefix; document COCO-image dependency)*
- **[fann_or_flop](./fann_or_flop/REPORT.md)** (medium-high) — *patch_with_metric_registration (register `FannOrFlopMetric` in `registry_metrics.yaml` so chrF++/BERTScore/judge stats a…*
- **[fig_qa](./fig_qa/REPORT.md)** (high) — *patch_with_use_validation_as_test_default*
- **[future_ideas](./future_ideas/REPORT.md)** (High) — *Before running, (1) author and pin the three judge rubrics, (2) implement a metric class or wire to a generic LLM-judge …*
- **[futuregen](./futuregen/REPORT.md)** (high) — *keep_as_is_with_caveats (optional patch: add LLM-judge metrics for novelty + feasibility to recover the paper's primary…*
- **[grapheval_review_advisor](./grapheval_review_advisor/REPORT.md)** (high) — *keep_as_is (with documented caveat: dataset is in repo but not discussed in the GraphEval paper primary tables; class-d…*
- **[hummus](./hummus/REPORT.md)** (high) — *patch_with_task_specific_metrics*
- **[humor_transfer](./humor_transfer/REPORT.md)** (high) — *keep_as_is_with_caveat (document the 2-of-4-subset scope reduction explicitly; optionally add `one_liners` and `dad_joke…*
- **[hypobench](./hypobench/REPORT.md)** (high) — *keep_as_is_with_synthetic_followup (document HDR/synthetic gap; build separate `hypobench_synthetic` scenario before pro…*
- **[ii_bench](./ii_bench/REPORT.md)** (high) — *keep_as_is_with_disclosure (use dev split N=35; submit to EvalAI for true test numbers if needed)*
- **[lcc_metaphor](./lcc_metaphor/REPORT.md)** (high) — *keep_as_is (note prompt is benchmark-authored; assert split count guard)*
- **[liveideabench](./liveideabench/REPORT.md)** (high) — *keep_as_is_with_caveat (single-judge substitution acceptable; reconcile stale registry rubric line vs. annotator)*
- **[meta4xnli](./meta4xnli/REPORT.md)** (high) — *keep_as_is (optional: add IV/OOV breakdown for detection)*
- **[metaphor_generation](./metaphor_generation/REPORT.md)** (high) — *keep_as_is_with_caveats (auto-eval slice only; human-eval out of scope)*
- **[moh_x](./moh_x/REPORT.md)** (high) — *patch_with_f1_score (add `f1_score` MetricSpec to run_spec; consider positive-class F1 to mirror Gao 2018)*
- **[munch](./munch/REPORT.md)** (high) — *keep_as_is_with_caveats (optionally add Metaphor-Sent + multi-prompt averaging from paper's prompts.md)*
- **[newyorker_humor](./newyorker_humor/REPORT.md)** (high) — *keep_as_is_with_caveats (matching/ranking are well-aligned; explanation task uses generation against gold-explanation re…*
- **[nyt_connections](./nyt_connections/REPORT.md)** (medium-high) — *keep_as_is_with_minor_fixes (note prompt provenance, set T=0.5, document instance scope)*
- **[ocw_connections](./ocw_connections/REPORT.md)** (high) — *patch_with_per_group_metric_aggregation (and T=0.0)*
- **[outline_to_story](./outline_to_story/REPORT.md)** (medium-high) — *keep_as_is_with_caveat (document the RAKE-side outline extraction; flag missing WikiPlots and perplexity)*
- **[poetmt](./poetmt/REPORT.md)** (high) — *patch_with_judge_alignment_and_comet (reconcile registry-vs-annotator judge, add COMET/BLEURT, fix instance-count docstr…*
- **[pron_vs_prompt](./pron_vs_prompt/REPORT.md)** (medium-high) — *keep_as_is_with_judge_substitution_caveat (reconcile registry vs implementation: registry advertises 3 judge metrics wit…*
- **[proparalogy](./proparalogy/REPORT.md)** (high) — *keep_as_is (positional bias bug has been fixed)*
- **[pun_eval](./pun_eval/REPORT.md)** (high) — *keep_as_is_with_caveats (or implement TPR/TNR/Kappa for full coverage)*
- **[puntuguese](./puntuguese/REPORT.md)** (high) — *keep_as_is_with_caveat (pun-location intentionally omitted; zero-shot prompted protocol is a documented shift from paper…*
- **[puzzleworld](./puzzleworld/REPORT.md)** (high) — *keep_as_is_with_caveats (final-answer accuracy only; document stepwise reasoning eval gap; consider stratified sampling)*
- **[rpgbench](./rpgbench/REPORT.md)** (medium-high) — *patch_with_validity_BFS_and_verbatim_judge_prompt (the current `json_validity` is JSON-parse only; paper's "validity" in…*
- **[schnovel](./schnovel/REPORT.md)** (high) — *patch_with_position_shuffle_lower_T_and_explicit_accuracy*
- **[science_analogies](./science_analogies/REPORT.md)** (high) — *patch_with_BLEURT (optional)*
- **[sdat](./sdat/REPORT.md)** (high) — *keep_as_is (English-only scope; document externally-recovered calibration constants; multilingual extension would lift t…*
- **[showerthoughts](./showerthoughts/REPORT.md)** (high) — *keep_as_is (linguistic/detector metrics from paper out of scope; populate the 6 judge prompts with paper's Likert rubric…*
- **[story_generation_rocstories](./story_generation_rocstories/REPORT.md)** (high) — *discuss (rebrand or add five-dimension judge)*
- **[story_quality](./story_quality/REPORT.md)** (medium) — *keep_as_is_with_caveat (correlation-metric bug now fixed; remaining deviations are task re-cast and reconstructed prompt…*
- **[thenextchapter](./thenextchapter/REPORT.md)** (high) — *keep_as_is_with_caveats (document metric-set differences vs paper; populate `judge_prompt` placeholders; consider robert…*
- **[tinystories](./tinystories/REPORT.md)** (high) — *keep_as_is (judge-model swap is benchmark-wide standardization; populate the 4 judge prompts with paper's grammar/creati…*
- **[ttcw](./ttcw/REPORT.md)** (medium) — *keep_as_is (fix paper pointer in registry; document N=36 open-text scope reduction)*
- **[twistlist](./twistlist/REPORT.md)** (high) — *patch_with_phonetic_metrics (PO/Init-PO/iPED/oPED — paper's signature contribution)*
- **[unfun_corpus](./unfun_corpus/REPORT.md)** (high) — *keep_as_is (classifier + human eval out of scope)*
- **[yesbut](./yesbut/REPORT.md)** (high) — *keep_as_is_with_caveat (Detection + Completion tasks intentionally out of scope; Understanding slice is faithful)*

## Tier 3 — retained as proxy / adapted (see sensitivity check)

- **[cpers](./cpers/REPORT.md)** (high) — *discuss (rework or drop)*
- **[creatset](./creatset/REPORT.md)** (high) — *discuss / patch_with_pairwise_winrate (drop BLEU-4 + ROUGE-L, replace with judge-based pairwise win-rate against a basel…*
- **[crowd_vote](./crowd_vote/REPORT.md)** (high) — *discuss / rename (rebrand as "marketing_creativity" — original benchmark is pairwise crowd voting; this is single-LLM-ju…*
- **[hypogen](./hypogen/REPORT.md)** (high) — *discuss / keep_as_proxy (already rebranded; document explicitly that this is NOT Si et al. 2024 and is not paper-compara…*
- **[mars](./mars/REPORT.md)** (medium-high) — *patch_with_quasi_em_or_ranking_protocol (replace exact_match with Hits@1/MRR-style or quasi-EM; current setup is structu…*
- **[metaphoric_analogies](./metaphoric_analogies/REPORT.md)** (high) — *patch_with_few_shot_and_lemmatized_metric (add few-shot examples; replace token-F1 with lemmatized head-noun matching; c…*
- **[pollux_creativity](./pollux_creativity/REPORT.md)** (high) — *discuss (fix scale 0-4 vs 0-1; populate per-criterion judge prompts; clarify metric structure)*
- **[scimon](./scimon/REPORT.md)** (high) — *patch_with_novelty_judge_or_rebrand (either add a novelty/relevance/technical-depth LLM-judge metric to recover the pape…*
- **[slang_generation](./slang_generation/REPORT.md)** (high) — *discuss (manifest-promised judges absent; metric is custom proxy not paper's suite; consider dropping or fully re-implem…*

## Notes

The italicized text under each benchmark is the per-bench audit
recommendation. For the live MetricSpec / AnnotatorSpec / ScenarioSpec
wiring HELM evaluates, see the implementation note at the top of each
report and the corresponding `run_specs/<bench>_run_specs.py`.

See [../../release_data/SCORING_NOTES.md](../../release_data/SCORING_NOTES.md) for the release-level scoring conventions (run-spec authoritative / registry descriptive, JRT 3-vendor panel for the 24 LLM-judge benchmarks, embedder_factory routing, Tier-3 sensitivity).
