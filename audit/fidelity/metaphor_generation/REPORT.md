# metaphor_generation fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (auto-eval slice only; human-eval out of scope)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/metaphor_generation_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Chakrabarty et al. (2021), MERMAID: Metaphor Generation with Symbolism and Discriminative Decoding (NAACL-2021)
- Repo: tuhinjubcse/MetaphorGenNAACL2021

## Implementation audited
- scenarios/metaphor_generation_scenario.py — zero-shot chat instruction "Rewrite the literal sentence below as exactly one metaphorical sentence... Output only the rewritten sentence." with `Literal sentence: ... \nMetaphorical sentence:` template. T=0.7, max_tokens=48, stop=["\n"].
- run_specs/metaphor_generation_run_specs.py — uses HELM `BasicGenerationMetric` with `bleu_4`, `rouge_l`, `f1_score`.
- Test set: `human1test.txt` + `human2test.txt` = 156 paired sentences. Verified.

## Deviations found
- [MEDIUM] **Original task** was seq2seq with symbolism/discriminative decoding rather than prompting an LLM. Surface form differs but input/output contract matches.
- [MEDIUM] **Human study not reproduced**: paper's pairwise human eval (metaphoricity, meaning preservation, creativity) is intentionally omitted. Acknowledged in metric_notes.
- [MEDIUM] **SBERT/BERTScore not implemented**.
- [LOW] `rouge_l`/`f1_score` are added diagnostics not in paper; `bleu_4` approximates upstream `score.py`.
- [LOW] `<V>` verb tags stripped before scoring (reasonable since BLEU/ROUGE work on surface tokens).

## Notes
Reference is single human-written metaphorical rewrite per item. No LLM judge wired. Tier 2: HIGH fidelity on automatic-eval slice, LOW on human-eval slice (intentionally out of scope). Results should be read as overlap diagnostics, not reproductions of MERMAID's primary human-preference claims.
