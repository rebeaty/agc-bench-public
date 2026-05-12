# Datasheet for AGC-Bench

Following the format of Gebru et al. (2018), *Datasheets for Datasets*. This datasheet describes the AGC-Bench release as a whole: the curated benchmark catalog, the runnable evaluation harness, the model-output corpus, and the validation analyses. Each constituent benchmark in the catalog has its own source-paper datasheet, cited inline in the per-benchmark Croissant metadata and the catalog CSV.

For per-benchmark paper-vs-implementation deviations (judge swaps, embedding-backend routing, prompt paraphrasing, partial-coverage caveats), see the fidelity audit at `audit/fidelity/`. For release-level scoring conventions and reproducibility scope, see `release_data/SCORING_NOTES.md`. For the data-quality sweep that produced the 32-cell release mask, see `audit/dq_sweep/`. For verbatim text of every LLM-judge prompt used (per-benchmark scoring rubrics, the on-task audit, the 3-LLM domain classifier, the AGC-Human fairness-aware judge, intervention prompts, MuCE judgment), see `audit/judge_prompts/`.

---

## Motivation

**For what purpose was the dataset created?**
AGC-Bench consolidates the published creativity-evaluation literature into a single calibrated harness, supporting cross-model questions about whether creativity in LLMs is general or domain-specific, separable from fluid reasoning, and responsive to creativity-targeted interventions.

**Who created the dataset and on behalf of which entity?**
[Anonymized for review. Camera-ready will list the lab, institution, and funding sources.]

**Who funded the creation of the dataset?**
[Anonymized for review.]

---

## Composition

**What do the instances that comprise the dataset represent?**
AGC-Bench is a meta-dataset with four artifact tiers:

1. **Benchmark catalog** — 497 unique creativity benchmarks identified through a PRISMA-compliant systematic review of 3,101 candidate papers (2018–2025). Each catalog entry is annotated with modality, scoring protocol, source paper, license, and a per-benchmark domain assignment.
2. **Evaluation harness** — runnable HELM-style scenarios for the 78 onboarded benchmarks in the release set (67 text-only, 11 multimodal), with 14 additional benchmarks documented as excluded for scope, methodology, or implementation-scope reasons (paper §3.1 Appendix). Each scenario carries the source paper's prompt template, generation configuration, and a HELM-compatible implementation of the source paper's canonical scoring procedure. Where the implementation deviates substantively from the source paper's metric (e.g., a paper-canonical metric replaced with a defensible proxy), the deviation is documented per-benchmark in `audit/fidelity/<bench>/REPORT.md` and tier-classified Tier-2 (spirit preserved) or Tier-3 (proxy / adapted; sensitivity check in `release_data/SCORING_NOTES.md`).
3. **Model-output corpus** — artifacts at the 83-model release scope: per-(model, dataset) z-scores, per-domain composites, leaderboard, AGC-Judge per-item predictions, JRT-corrected ratings, and the derived item-level generation corpus hosted with the Hugging Face release.
4. **Validation analyses** — per-domain composites, c-factor loadings and parameters, intervention deltas (be-creative, reasoning on/off), MuCE judgment-prediction outputs, paired-human (CAP) scores, and the c-factor robustness checks.

**How many instances are there in total?**
- 497 unique benchmarks in the catalog (78 onboarded into the harness for this release).
- 5,561 possible (model, dataset) cells across the 83-model × 67-text-only primary analysis; 5,478 actually carry a canonical score in the released long table. The 83-cell gap reflects 51 cells the model never produced a canonical score for (concentrated on `irfl × 50`, a multimodal-only benchmark not required of the strict text-only release set) and 32 cells masked by the release data-quality sweep (`dq_masked = True`; see `audit/dq_sweep/REPORT.md`). Pre-mask, every release model had a canonical score on ≥ 65 of 67 datasets; after the mask, four release models drop below that floor — `google/gemma-2-27b-it` at 54, `morph/morph-v3-fast` at 57, `meta-llama/llama-3.2-3b-instruct` at 63, `z-ai/glm-4.5v` at 64 — with the rest distributed as 31 at 67, 43 at 66, 5 at 65.
- ~278,000 individual (model, dataset, item) generations across the primary release set.

**Does the dataset contain all possible instances or is it a sample?**
The catalog contains all unique benchmarks identified by the PRISMA review through December 2025; the onboarded subset (78 = 67 text-only + 11 multimodal) is the working frontier, with 14 additional benchmarks documented as excluded (paper §3.1 Appendix). Released score and generation artifacts cover the 83-model release set that supports the aggregate analyses.

**What data does each instance consist of?**
Each catalog entry: paper title, source-paper venue, year, DOI/URL, license, primary domain, modality tier, canonical metric, original task description, and a stable benchmark identifier. Each onboarded benchmark adds a runnable scenario file. The released tables store dataset-level z-scores, domain composites, AGC-Judge validation outputs, and the primary composite per model. The Hugging Face generation corpus stores release-set prompts, completions, and metric metadata where available; calibrated judge outputs are summarized in the released JRT and AGC-Judge artifacts.

**Is there a label or target associated with each instance?**
For closed-ended benchmarks, each item has a reference answer from the source paper. For open-ended generation benchmarks, the source paper's canonical scoring metric (LLM-judge rubric, semantic-distance metric, expert-rubric, etc.) is applied to each generation and serves as the target signal. Where a paper-canonical metric was replaced with a defensible proxy (Tier-3 in the fidelity audit), the substitution is documented per-benchmark in `audit/fidelity/<bench>/REPORT.md` and the primary-result sensitivity check in `release_data/SCORING_NOTES.md` reports the delta.

**Is any information missing from individual instances?**
Yes, by design. (a) A subset of primary-analysis datasets do not write per-instance scores in the current run logs (only dataset-level aggregates) — see Appendix C for the per-dataset list. (b) Some (model, dataset) cells do not carry a canonical score: 51 are coverage gaps (mostly `irfl × 50`, a multimodal-only benchmark) and 32 were masked by the release data-quality sweep when both heuristic flags (empty / refusal / repetitive responses) and the on-task LLM-judge audit identified the same cell as off-task. Per-model coverage distributes as 31 at 67 / 43 at 66 / 5 at 65, with four below the pre-mask floor (`gemma-2-27b-it` at 54, `morph-v3-fast` at 57, `llama-3.2-3b-instruct` at 63, `glm-4.5v` at 64). The `dq_masked` column in `release_data/long_model_x_dataset.csv` distinguishes mask from coverage gap.

**Are relationships between individual instances made explicit?**
Yes. Each (model, dataset, item) cell is linked to its source benchmark via the catalog identifier, to its source paper via DOI, and to the model via a versioned model identifier. The taxonomy file maps each dataset to its primary domain.

**Are there recommended data splits?**
The paper distinguishes two evaluation surfaces: the primary 67-dataset benchmark (used for c-factor, separability, MuCE judgment, leaderboard) and the AGC-Human paired-human subset (used for direct human–LLM comparison and intervention experiments). Each constituent benchmark retains its source paper's train/dev/test split where one exists; cell-level evaluation samples from the test split with a frozen seeded selection of n=50 items per cell.

**Are there any errors, sources of noise, or redundancies in the dataset?**
Several known sources of noise are documented:
- LLM-as-judge metrics inherit residual judge bias; we report a three-signal validation (κ=0.67 inter-judge, r=0.49 median per-dataset judge–metric convergence, adversarial smoke tests; see §3.4 of the paper).
- Text-generation cells with truncation: 16 datasets were re-run at raised token caps; `critics_story` remained over-budget at 8,192 tokens and was dropped from the primary analysis. `fann_or_flop` is retained in the primary release set despite verbose-tail truncation; per-dataset truncation rates are documented in Appendix C.2.
- Length confound on open-ended scoring: the CrPO scoring panel is reported in both raw and length-residualized forms.
- Source-paper deduplication: 33 duplicate groups were merged via 10-pass Gemini consensus voting (≥80% agreement); residual near-duplicates may exist.

**Is the dataset self-contained, or does it link to or rely on external resources?**
The harness, model-output corpus, and validation analyses are self-contained. The catalog links to source papers and source-paper-hosted data via DOI/URL; running individual benchmarks may require downloading source data from those external locations subject to per-benchmark licenses. The MuCE convergent-validity test relies on the MuCE benchmark (Ismayilzada et al., 2025), and the AGC-Human paired-human subset uses the Creativity Assessment Platform (Patterson et al., 2025).

**Does the dataset contain data that might be considered confidential?**
No. All catalog data is drawn from publicly available papers. Model generations are produced by publicly accessible APIs or open-weight models. CAP human-comparison data was collected under [Author institution] IRB [protocol number redacted for review] with anonymization at intake; no personally identifying information is released.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
The model-output corpus contains generations from the 83 release models across the primary AGC-Bench evaluation surface. Tasks include open-ended creative writing, humor, and ideation. Some benchmark items deliberately probe edge cases such as malevolent creativity (`Malevolent Problems` in MuCE) where models are asked to generate creative responses to ethically charged prompts. These are part of the source benchmarks' design and reflect the published evaluation literature; no synthetic adult, harmful, or NSFW content was generated outside that scope. A 95.1% on-task / 4.9% invalid rate from the data-quality audit (§3.4) characterizes the residual; 32 cells flagged by both that audit and the heuristic sweep are masked at the score level (`dq_masked = True` in the long table).

**Does the dataset relate to people?**
Indirectly through the AGC-Human paired-human subset (CAP), which contains anonymized human responses on five creativity tasks collected under [Author institution] IRB [protocol number redacted for review]. No individual humans are identifiable from the released data. The model-output corpus does not relate to people.

---

## Collection Process

**How was the data associated with each instance acquired?**
- *Catalog*: Semantic Scholar harvest (n=3,101 candidate papers), GPT-4.1 inclusion-rubric pre-screening (683 retained), Gemini 2.5 Flash data-availability verification (431 verified), four-reviewer dual human review (rotating assignment, lead-author conflict resolution), Gemini 2.5 Pro benchmark extraction (546 benchmarks across 1,160 tasks), and 10-pass Gemini consensus deduplication (497 unique benchmarks).
- *Onboarded scenarios*: staged onboarding workflow from paper parsing through HELM scenario implementation, with prompts extracted verbatim, scoring metrics preserved, and human review at every accept step (Appendix B).
- *Model-output corpus*: release-set evaluation against proprietary APIs and open-weight models served via OpenRouter, with reasoning disabled by default for release-set comparability and a matched-pairs reasoning-on/off subset retained for intervention analyses.
- *AGC-Human (CAP)*: anonymized human responses collected on the Creativity Assessment Platform under [Author institution] IRB [protocol number redacted for review].

**What mechanisms or procedures were used to collect the data?**
The curation pipeline is implemented as reproducible Python modules with checkpoint support and deterministic seeding. Onboarding uses a deterministic state machine and per-state validation. Release-set evaluation uses a HELM-style harness. Score artifacts (per-(model, dataset) z-scores, per-domain composites, leaderboard, AGC-Judge per-item predictions on the 24 LLM-judge benchmarks) and the derived release-set generation corpus ship with this release. The public curation and onboarding artifacts are included under `curation/`.

**If the dataset is a sample from a larger set, what was the sampling strategy?**
The 78 onboarded benchmarks are a curated subset of the 432 creativity-relevant candidates after tier classification. Tier-1a (text-to-text, standard API call) candidates entered onboarding; higher tiers were retained in the catalog for future releases. Per-cell evaluation uses a frozen seeded selection of n=50 items per dataset; rank-stability bootstrap analyses (§3.4) confirm Spearman ρ > 0.99 with k=10 items.

**Who was involved in the data collection process and how were they compensated?**
Four team members performed the dual human review of 431 candidate papers as part of standard research duties. Human participants on the AGC-Human paired-human subset were recruited and compensated under the [Author institution] IRB [protocol number redacted for review] protocol.

**Over what timeframe was the data collected?**
- Catalog: literature harvest covers 2018–2025, with the systematic review conducted in 2025.
- Release-set evaluation: model evaluations conducted between 2024 and 2026, with the strict-coverage release set frozen at the submission cutoff.
- AGC-Human: human responses drawn from prior approved Creativity Assessment Platform (CAP) collections; earliest and latest CAP collection dates withheld for double-blind submission.

**Were any ethical review processes conducted?**
The AGC-Human paired-human subset was collected under [Author institution] IRB [protocol number redacted for review]. Catalog and release-set evaluations involved no human subjects and did not require additional IRB review.

---

## Preprocessing/Cleaning/Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes. (a) Each onboarded benchmark's source data was loaded via the harness's dataset loader, with stable item IDs assigned for reproducibility. (b) For each evaluated (model, dataset) cell, the canonical metric was computed from the model's generation; released per-instance records are provided in the generation corpus where available, and aggregate score artifacts are provided in `release_data/`. (c) Within each dataset, raw metric values are z-scored across the release set; per-(model, dataset) cells are then mean-aggregated when a dataset has multiple canonical metrics. (d) The AGC-Human subset uses the CrPO scoring panel (diversity, DSI, surprise) computed at the response level, with length-residualized variants for verbosity-confound control.

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**
Yes. The Hugging Face release includes the derived release-set generation corpus, prompt banks, and consolidation audit. The released bundle also contains score artifacts: dataset-level z-scores, per-(model, dataset) cells in `release_data/long_model_x_dataset.csv`, the per-(model, item) AGC-Judge prediction file for the 24 LLM-judge benchmarks, JRT-corrected ratings, and the leaderboard.

**Is the software used to preprocess/clean/label the instances available?**
Yes. The harness, scoring code, judge prompts, and aggregation scripts are released under Apache 2.0 (code) and CC BY 4.0 (data outputs); see LICENSE and LICENSE-DATA.

---

## Uses

**Has the dataset been used for any tasks already?**
The dataset has been used for the four validation analyses reported in the AGC-Bench paper:
1. C-factor extraction across 6 text-only domains under the v3 LLM-panel partition (single-factor unidimensional structure with α=0.96, 81.5% variance, parallel-analysis confirmed; all six domain loadings between +0.87 and +0.94).
2. Separability of c from general capability (LSA pure-Gf r=+0.53 Pearson; MMLU-Pro purer-Gc r=+0.63; the two indicators are statistically indistinguishable, matching Gerwig 2021's human meta-analytic pattern; c × MuCE judgment partial r=+0.34 after partialling out both LSA and GPQA-Diamond jointly).
3. Intervention sensitivity (be-creative dz=+1.75 unadjusted, +1.40 length-residualized vs. reasoning-on dz=+0.34 length-residualized; reasoning-on dissociation: diversity improves at dz=+0.65, novelty barely moves at dz=-0.01).
4. Paired human–LLM comparison on five CAP tasks (top human reaches +0.65 composite vs top LLM moonshotai/kimi-k2-0905 at +0.53; humans dominate the upper tail with 22 of the top 30 entities). Within-instrument bootstrap shows LLMs are more domain-general than humans on every standard psychometric indicator (Cronbach α=0.64 vs 0.42, all four bootstrap p in [0.005, 0.028]).

**Is there a repository that links to any or all papers or systems that use the dataset?**
The dataset and citation tracker live at https://huggingface.co/datasets/agcbench-2026/AGC-Bench; the AGC-Judge scorer at https://huggingface.co/agcbench-2026/AGC-Judge.

**What (other) tasks could the dataset be used for?**
Beyond the analyses in the paper, AGC-Bench supports:
- Studies of mechanism: tracing how training data, architecture, and post-training shape c.
- Studies of enhancement: prompting, fine-tuning, decoding strategy, retrieval, and multi-step evaluation scaffolds.
- New-model evaluation: any newly released LLM can be added to the harness and ranked against the same panel without re-engineering.
- Benchmark-level analyses: per-benchmark difficulty, discrimination, item-level signals, and contributions to the c-factor.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
Yes:
- *English-dominant scope.* This release is predominantly English; multilingual extension is left to future work.
- *Image-input multimodal only.* Video, audio, and image-output benchmarks are out of scope for this release.
- *Release-set-relative composite.* The primary composite is a release-set z-score and has no theoretical ceiling on the open-ended subset; rankings shift if the release set changes.
- *Source-benchmark licenses.* Each onboarded benchmark retains its source-paper license; some preclude commercial use or redistribution.
- *Judge-model dependence.* Several canonical metrics use LLM-as-judge scoring with provider-specific judge models; downstream uses should re-run the audit if the judge endpoint changes.

**Are there tasks for which the dataset should not be used?**
- The benchmark should not be used to train models on the underlying source-benchmark items where source paper licenses preclude such use.
- The release-set-relative composite should not be interpreted as an absolute creativity score; the closed-ended subset has a 100% ceiling and supports absolute claims only for that slice.
- The Malevolent Problems items in the MuCE convergent-validity test are designed to probe ethically charged ideation; the data-quality audit log is the appropriate reference for what was filtered.

---

## Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**
Yes. AGC-Bench is released publicly.

**How will the dataset be distributed?**
Hosted on HuggingFace at https://huggingface.co/datasets/agcbench-2026/AGC-Bench, paired with the AGC-Judge model release at https://huggingface.co/agcbench-2026/AGC-Judge. The release contains:
- Benchmark catalog: Croissant 1.1 JSON-LD manifest + per-benchmark CSV.
- Runnable harness: HELM-style scenarios, run specs, scoring code.
- Aggregate score artifacts: per-(model, dataset) z-scores, JRT-corrected ratings, AGC-Judge per-item predictions, leaderboard.
- Validation analyses: c-factor loadings, intelligence correlations, intervention deltas, AGC-Human composites.

**When will the dataset be distributed?**
Initial release: NeurIPS 2026 D&B submission package. Public archival release: at the camera-ready deadline if accepted, or at a follow-up rebuild date if not. Persistence commitment per NeurIPS D&B requirements.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license?**
The harness, scoring code, aggregation scripts, and analysis outputs are released under Apache 2.0 (code) and CC BY 4.0 (data: release_data/, analysis/, croissant/, registry, datasheet). Each constituent benchmark retains its source-paper license, documented in the catalog's per-benchmark license column. Users running individual benchmarks must comply with the source paper's license.

**Have any third parties imposed IP-based or other restrictions on the data?**
Per-benchmark licenses imposed by the source papers — see the catalog's license column. The 11 multimodal benchmarks include some with research-only licenses; consult the per-benchmark Croissant.

**Do any export controls or other regulatory restrictions apply to the dataset?**
None known.

---

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
[Anonymized for review. Camera-ready will name the lab and contact.]

**How can the owner/curator/manager of the dataset be contacted?**
[Anonymized for review.]

**Is there an erratum?**
None at this time. Package updates are listed in `CHANGELOG.md`.

**Will the dataset be updated?**
Yes. Planned updates:
- *Catalog expansion.* The 78 onboarded benchmarks are the current frontier; the harness is designed to extend to the full 432-candidate catalog as additional benchmarks are integrated.
- *New-model additions.* The leaderboard accepts new model submissions; rankings are recomputed against the frozen release set.
- *Multilingual extension.* Non-English creativity benchmarks are left to future work.
- *Multimodal extension.* Video, audio, and image-output benchmarks are left to future work.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**
The AGC-Human subset retains anonymized human responses under the terms of [Author institution] IRB [protocol number redacted for review] with no time-limited retention restriction beyond the standard institutional record-keeping rules.

**Will older versions of the dataset continue to be supported/hosted/maintained?**
Yes. Each released version is tagged on HuggingFace; older releases remain accessible alongside the latest.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes. Contributions are accepted via pull requests on the release repository. New benchmarks added through the onboarding pipeline pass the same dual-review and pilot-execution checks as the released set. New model evaluations can be submitted via the leaderboard intake form.
