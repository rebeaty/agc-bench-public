# unfun_corpus fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (classifier + human eval out of scope)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/unfun_corpus_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.unfun_automatic_metric.UnfunAutomaticMetric`
> - **ScenarioSpec args:** `prompt_style='chat'`, `context_size=8`, `seed=1234`, `include_validation=False`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Horvitz et al. 2024, "Getting Serious about Humor" (arXiv:2403.00794, ACL 2024)
- Repo: github.com/zacharyhorvitz/Getting-Serious-With-LLMs

## Implementation audited
- scenarios/unfun_corpus_scenario.py — builds 8-shot in-context prompt sampled from `train_for_prompting.tsv` with deterministic per-instance seeding, mirroring upstream `hit_llm_generation_v2.py` and `data_generation/prompts/unfun_dataset/few-shot/`.
- Two styles: `chat` (default; system message + alternating User/Assistant turns) and `completion` (`headline -> unfunned` line format). Both reproduce upstream wording verbatim.
- run_specs/unfun_corpus_run_specs.py — `prompt_style="chat"`, `context_size=8`, `seed=1234`.
- metrics/unfun_automatic_metric.py — `UnfunAutomaticMetric` computes (a) mean token-level edit distance from satirical source after lowercasing + punctuation stripping, (b) corpus-level type-token ratio. Logic matches upstream `_find_edit_distance`.
- 375 test instances (paper's no-leakage held-out split); optional 186 validation off.

## Deviations found
- [MEDIUM] **Classifier accuracy not implemented**: paper reports humor-classifier holdout accuracy (RoBERTa). Acceptable scope reduction.
- [MEDIUM] **Human ratings not implemented**: paper's funniness, realness, grammaticality 1-5 ratings absent.
- [LOW] Inference defaults (T=0.0, max_tokens=64, stop=`\n`); paper unspecified.

## Notes
Reference = human-authored `unfunned_headline` from Unfun game corpus, attached as CORRECT_TAG. No LLM judge — metrics are reference-free formulas over generation + satirical source. Faithful prompt construction, splits, and automatic metrics. Tier 2 due to scope reduction (classifier and human eval omitted).
