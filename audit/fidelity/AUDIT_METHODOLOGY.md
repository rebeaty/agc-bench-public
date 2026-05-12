# AGC-Bench v1 fidelity audit — methodology

This document specifies the protocol the per-benchmark fidelity audits in
`audit/fidelity/<bench>/REPORT.md` follow. The current audit covers all 78
v1 benchmarks; tier distribution and per-benchmark recommendations live in
[SUMMARY.md](./SUMMARY.md).

---

## What an audit checks

Each report answers one question per benchmark: **does the AGC-Bench
implementation faithfully reproduce what the source paper specifies, within
HELM-harness tolerance?**

Four axes are evaluated:

1. **Instance source** — Is the loaded data the same dataset the paper used?
   Same split? Same n? Any subsampling, filtering, or transformation?
2. **Prompt fidelity** — Is the model-facing prompt verbatim from the paper,
   or paraphrased? If paraphrased, does the spirit match?
3. **Metric / scoring fidelity** — Does the metric we register match the
   paper's primary metric? If we substituted (e.g., GPT-4 judge → Gemini
   judge, embedding model swap), is the substitution within HELM-wide
   standardization tolerance?
4. **Generation configuration** — Temperature, max-tokens, n-samples — match
   the paper?

Each axis can deviate at three levels:

- **HIGH** — structurally wrong (broken metric, hardcoded answer position,
  wrong dataset). Blocks paper-comparability.
- **MEDIUM** — substantive but defensible (different judge model, partial
  subtask coverage, paraphrased prompt with same intent).
- **LOW** — cosmetic or HELM-standardization (judge swap to Gemini for release-set
  comparability, metric variant, fewer n-samples).

---

## Tier classification

Severity bubbles up to a benchmark-level tier:

- **Tier 1** — faithful within HELM tolerance (only LOW deviations on the
  four axes). Keep as-is.
- **Tier 2** — notable deviation, spirit preserved. May include HIGH
  deviations on a single axis (most commonly metric/scoring fidelity, where
  a paper-canonical metric was substituted with a defensible analog or where
  one of multiple metric variants couldn't be implemented). The implemented
  evaluation produces a coherent signal aligned with the source paper. Action:
  keep as-is with caveat, patch, or document.
- **Tier 3** — structural defect. HIGH deviations on multiple axes, or a
  HIGH on the metric/scoring axis where the implemented metric differs in
  substantive construct from the paper's primary metric (a defensible
  proxy or adapted scoring path, not the paper's exact protocol). v1
  retains Tier-3 benchmarks in the primary release set with the deviation
  documented per-bench; readers preferring strict exclusion can read
  the "drop 8 Tier-3" sensitivity column in
  [`../../release_data/SCORING_NOTES.md`](../../release_data/SCORING_NOTES.md). Primary
  intelligence ρ values shift by ≤ 0.01 under that exclusion; c-factor
  magnitude shrinks (eigenvalue 4.89 → 4.22, variance 81.5 % → 70.4 %)
  but stays unidimensional with α = 0.91.
- **Tier ?** — insufficient evidence to classify; needs manual review.

The Tier 2 / Tier 3 boundary is a judgment call. The question is whether the
produced score is interpretable as the paper's intended construct, even if
the literal metric machinery differs. Reports document the reasoning per
benchmark.

---

## Authority order: run_specs canonical, registry descriptive

A small number of per-benchmark reports describe `data/registry/registry_metrics.yaml`
as the canonical metric surface. The registry catalogues each source paper's
**available** metric surface (BLEU, ROUGE, embedding models, candidate
LLM-judge entries) for documentation. What the released runs **actually
computed** is determined by the corresponding `run_specs/<bench>_run_specs.py`,
which wires specific MetricSpec and AnnotatorSpec instances into HELM. Where
the two diverge, the run-spec is authoritative.

For benchmarks whose body describes a registry-only metric layout that has
since been wired into the run-spec, an "Implementation note" at the top of
the per-benchmark REPORT.md names the live MetricSpec / AnnotatorSpec /
ScenarioSpec wiring (37 of 78 reports). For benchmarks without such a note,
the report body and `run_specs/<bench>_run_specs.py` are the canonical
reference. See `release_data/SCORING_NOTES.md` for the release-level authority story.

---

## What this audit does NOT cover

- **Score quality / model performance.** The audit checks paper-vs-implementation
  fidelity, not whether the resulting numbers are good or bad.
- **Inter-judge agreement.** Inter-judge κ and JRT calibration are reported
  in §3.5 of the paper and `audit/dq_sweep/` does not duplicate them.
- **Per-cell data quality.** The cell-level on-task / garbled audit is in
  `audit/dq_sweep/`, separate from this fidelity audit.

---

## Provenance

Reports were authored as one-shot Claude reviews per benchmark, drawing on
the implementation files (scenario, run_spec, registry entries, metric module,
annotator) plus the source paper / repository. Where the report body
describes a registry-only metric layout that has since been wired into the
run-spec, an "Implementation note" at the top names the live wiring; the
deviation list below the note describes the paper-vs-implementation gap.
