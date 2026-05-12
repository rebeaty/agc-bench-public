# historical_analogy fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (optional: add general-analogy split + MDS judge)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/historical_analogy_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.historical_analogy_metric.HistoricalAnalogyMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: "Past Meets Present: Creating Historical Analogy with LLMs" (ACL 2025; arXiv 2409.14820)
- Repo: Nianqi-Li/Historical-Analogy-of-LLMs

## Implementation audited
- scenarios/historical_analogy_scenario.py — reproduces `direct_generation.py`'s 1-shot template verbatim: bot framing, `==== case` block with COVID-19 → Spanish flu exemplar, then test event + intro and trailing `Historical Analogies Events:` cue.
- metrics/historical_analogy_metric.py (`HistoricalAnalogyMetric`) — queries Wikipedia OpenSearch for predicted and gold titles and checks set intersection, mirroring upstream `evaluation.py`. Adds auxiliary `parsed_event_rate` diagnostic.
- 20 popular analogies from `popular_analogy.jsonl` — matches paper's curated popular set.

## Deviations found
- [MEDIUM] **MDS metric omitted**: paper has two evaluation tracks: (a) Pass@1 on popular analogies via Wikipedia title overlap, and (b) Multi-dimensional Similarity (MDS) for general analogies plus human ranking. HELM implements only (a).
- [MEDIUM] **General-analogy split not loaded** (~50 items).
- [INFO] Inference uses HELM defaults (greedy, newline stop) — reasonable approximation of repo's deterministic decoding.

## Notes
Reference = `target_event` from dataset; "judge" is Wikipedia OpenSearch API (deterministic, same as repo). No LLM judge invoked — consistent with repo's Pass@1 path. Tier 1 because the popular-analogy Pass@1 evaluation faithfully reproduces paper's primary quantitative metric, including the Wikipedia-search matching idiosyncrasy.

Limitation: MDS LLM-judge metric and general-analogy split are absent, so coverage is the closed-form half of the paper's evaluation only. Optional extension with `general_analogy.jsonl` + MDS judge would cover the divergent-thinking dimension emphasized in paper.
