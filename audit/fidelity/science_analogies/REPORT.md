# science_analogies fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_BLEURT (optional)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/science_analogies_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.science_analogies_metric.ScienceAnalogiesAutomaticMetric`
> - **ScenarioSpec args:** `subset=subset`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Bhavya et al., "Analogy Generation by Prompting Large Language Models: A Case Study of InstructGPT" (INLG 2022, arXiv:2210.04186)
- Repo: github.com/Bhaavya/InstructGPT-Analogies

## Implementation audited
- scenarios/science_analogies_scenario.py — uses verbatim templates from upstream `plm_generator.py`:
  - `nosrc`: `"Explain {target} using an analogy."`
  - `wsrc`: `"Explain {target} using an analogy involving {source}."`
- run_specs: zero-shot, T=0.0, max_tokens=256, stop `\n\n`.
- metrics/science_analogies_metric.py — drops BLEURT (scorer not packaged), substitutes BLEU-4. Preserves ROUGE-L and METEOR.
- 109 nosrc unique target concepts + 148 wsrc triples — matches paper exactly.

## Deviations found
- [HIGH] **BLEURT not implemented**: paper's primary automatic metric. Scorer not packaged in HELM stack.
- [HIGH] **Human qualitative ratings not implemented**: paper's meaningfulness, originality, common-sense ratings absent.
- [LOW] **BLEU-4 substitution**: substituted as a lexical diagnostic; paper does not use BLEU.
- [LOW] **Greedy T=0** vs paper's unspecified decoding (acceptable for benchmarking).
- [LOW] `avg_response_words` length diagnostic added (not in paper).

## Notes
References are gold explanations from saqa.txt (1–4 per nosrc target; 1 per wsrc instance), tagged CORRECT_TAG. No judge model. Tier 2 because BLEURT (paper's primary automatic metric) is missing and qualitative scoring is absent — leaves only n-gram/word-overlap signals which paper itself flags as weak proxies for analogy quality.
