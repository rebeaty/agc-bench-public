# AGC-Bench

AGC-Bench (Artificial General Creativity Benchmark) is a HELM-compatible evaluation suite for measuring creative ability in language and vision-language models. The release includes runnable benchmark scenarios, scoring code, release tables, and scripts that reproduce the paper's **83-model** leaderboard. It covers **78 datasets**: **67 text-only benchmarks** in the primary analysis plus **11 multimodal-only** scenarios released as artifacts. Domains span Brainstorming, Problem Solving, STEM, Story / Narrative, Figurative Language, and Humor. See [DATASHEET.md](DATASHEET.md) for the public dataset card.

**Interactive leaderboard:** open [agcbench-2026/agc-bench-leaderboard](https://huggingface.co/spaces/agcbench-2026/agc-bench-leaderboard) for sortable composite and per-domain rankings. The Space is generated from [release_data/leaderboard.csv](release_data/leaderboard.csv) and [analysis/per_domain_jrt.csv](analysis/per_domain_jrt.csv).

**Release note:** v1.0.1 adds the 497-row benchmark catalog, a compact Dataset Viewer table, and clearer maps of the repository contents. Scores and validation artifacts are unchanged. See [CHANGELOG.md](CHANGELOG.md).

**Start here:** [release_data/README.md](release_data/README.md) maps the shipped score tables, [analysis/README.md](analysis/README.md) maps validation artifacts, [audit/README.md](audit/README.md) maps audit reports, and [scripts/README.md](scripts/README.md) maps the reproduction pipeline.

---

## Overview

Each dataset is implemented as a HELM [Scenario](https://github.com/stanford-crfm/helm) plus a `RunSpec`. The `RunSpec` defines the evaluation HELM executes; the registry files document source-paper metrics and inference settings. Cross-benchmark scoring notes are in [release_data/SCORING_NOTES.md](release_data/SCORING_NOTES.md), and per-benchmark implementation audits are in [audit/fidelity/](audit/fidelity/).

| Path | Purpose |
|---|---|
| [scenarios/](scenarios/) | HELM dataset loaders: one `Scenario` per benchmark. |
| [run_specs/](run_specs/) | HELM run definitions: metrics, annotators, and adapter settings. |
| [eval_scripts/](eval_scripts/) | Shell entry points for running one benchmark, all benchmarks, or a new-model AGC-Judge run. |
| [metrics/](metrics/) | Metric implementations used by the run specs. |
| [llm_judge/](llm_judge/) | LLM-judge annotators, judge metrics, and the AGC-Judge endpoint client. |
| [clients/](clients/) | Custom HELM clients for provider-specific routing. |
| [prod_env/](prod_env/) | Deployment templates and model routing configuration. |
| [data/](data/) | Small bundled inputs and source-paper registry files; not the main score release. |
| [release_data/](release_data/) | Frozen score tables, leaderboard, benchmark catalog, and companion release tables. |
| [generations/](generations/) | Hugging Face-hosted model-output corpus: release-set prompts, completions, and consolidation audit. |
| [analysis/](analysis/) | Validation artifacts: JRT ratings, c-factor loadings, intelligence joins, domain panels, and external checks. |
| [audit/](audit/) | Release audit materials: implementation fidelity, data-quality sweep, prompt appendix, and supporting reports. |
| [scripts/](scripts/) | Rebuild and analysis scripts for the released artifacts. |
| [croissant/](croissant/) | Croissant 1.1 manifests (aggregate plus per-benchmark). |
| [curation/](curation/) | Catalog construction and benchmark-onboarding records. |

---

## Reproducibility Quickstart

From the bundle root, the released leaderboard and paper claims can be checked from bundled artifacts without running model inference, serving AGC-Judge, or using a GPU:

```bash
bash reproduce_paper_results.sh
bash reproduce_appendix.sh
```

`reproduce_paper_results.sh` checks the main-text numerical claims against the 83 strict-coverage release models. `reproduce_appendix.sh` checks a panel of appendix claims and reports PASS / WITHIN_TOL / FAIL status for each check. Both scripts print the paper-stated and recomputed values.

If starting from a fresh Python environment, install the package first with `pip install -e ".[eval,dev]"` as shown in [Setup](#setup).

Reproducibility surfaces:

| Goal | Command / entry point | Requires model calls? | Requires AGC-Judge serving? |
|---|---|---:|---:|
| Main paper claims | `bash reproduce_paper_results.sh` | No | No |
| Appendix numerical checks | `bash reproduce_appendix.sh` | No | No |
| AGC-Judge reproduction of released model rankings | `python3 scripts/show_agc_judge_repro.py --all` | No | No |
| Score a new model | `bash eval_scripts/run_with_agc_judge.sh <model>` | Yes | Yes |

These checks do not re-query commercial models. They recompute analyses from the released `release_data/` and `analysis/` files, including the leaderboard and AGC-Judge held-out prediction artifacts.

For the per-script map (analysis pipeline, release-data rebuild, AGC-Judge external validation), see [scripts/README.md](scripts/README.md). For release scope and rounding notes, see [release_data/SCORING_NOTES.md](release_data/SCORING_NOTES.md).

## Check AGC-Judge Leaderboard Reproduction

```bash
python3 scripts/show_agc_judge_repro.py claude-opus-4.5
python3 scripts/show_agc_judge_repro.py --all
```

Prints AGC-Judge-only composite predictions versus the JRT-corrected released leaderboard for the in-distribution, held-out-models, and held-out-benches splits. This is a no-compute check of the released AGC-Judge validation artifacts, not a live inference run.

## Score a New Model End-to-End

```bash
bash eval_scripts/run_with_agc_judge.sh openai/gpt-5.5
```

This runs a new model on the 67 text-only primary AGC-Bench datasets through the released HELM pipeline. The 43 non-JRT datasets use their implemented canonical metrics, including formula, embedding, and model-based metrics. The 24 JRT-corrected LLM-judge datasets route judge calls through **AGC-Judge**, the released Qwen3-30B-A3B-Instruct-2507 LoRA fine-tune, via `AGC_JUDGE_OVERRIDE`.

After completion: `analysis/scored/<model>/` contains the leaderboard slot
(`leaderboard_line.json`), per-domain mean-z (`per_domain.csv`), per-bench
z-normalized scores (`per_dataset_z.csv`), and any datasets HELM did not
produce a `stats.json` for (`missing.csv`, candidates for re-run).

Full new-model scoring requires external resources:

- Access to the model under evaluation, typically through OpenRouter.
- A running AGC-Judge endpoint, either in your HF account or on local GPU hardware.
- An embedding backend for the nine embedding-driven metrics. The default script uses local Qwen embeddings; exact published-cohort parity on those metrics uses Gemini embeddings.

### AGC-Judge Serving

AGC-Judge is published on HF as a LoRA adapter at
[huggingface.co/agcbench-2026/AGC-Judge](https://huggingface.co/agcbench-2026/AGC-Judge).

Option A: deploy AGC-Judge with HF Inference Endpoints in your own HF account:

```bash
huggingface-cli login
python eval_scripts/spin_up_hf_endpoint.py
```

The script asks for confirmation before creating an endpoint, deploys under your HF namespace by default, polls until the endpoint is running, and writes the endpoint URL into `prod_env/model_deployments.yaml`. HF pricing and quota are account- and region-dependent; check the HF Endpoint console before confirming creation. The endpoint is configured with scale-to-zero. To delete it entirely:

```bash
python eval_scripts/tear_down_hf_endpoint.py --delete
```

Option B: serve AGC-Judge locally with vLLM on an 80GB+ GPU:

```bash
bash eval_scripts/run_with_agc_judge.sh --local-vllm openai/gpt-5.5
```

`--local-vllm` starts vLLM, points HELM at `localhost:8000/v1` for judge calls, restores the previous deployment config on exit, and keeps the local Qwen embedding fallback and BERTScore-based metrics on CPU by default so they do not compete with AGC-Judge for VRAM. Override `AGC_QWEN_EMBEDDING_DEVICE` or `AGC_BERT_SCORE_DEVICE` if you have spare GPU capacity.

### Credentials

Copy the credentials template and fill in the keys needed for your chosen path:

```bash
cp prod_env/credentials.conf.template prod_env/credentials.conf
```

At minimum, new-model scoring usually needs `openrouterApiKey` for the model under evaluation. HF Endpoint serving additionally needs `huggingfaceApiToken`. Exact published-cohort embedding parity needs `googleApiKey` plus `AGC_EMBEDDING_BACKEND=gemini`.

### Embedding Backend

Nine metrics call out to an embedding model (sdat, conceptual_design,
slang_generation, mops_diversity, semantic_diversity, etc.; see
`metrics/embedder_factory.py` for the full list). The published cohort
used `gemini-embedding-001` for cross-bench comparability:

- `AGC_EMBEDDING_BACKEND=gemini` — paid Google API key required
  for exact published-cohort match. Quota and billing depend on the Google
  account used for the run.
- `AGC_EMBEDDING_BACKEND=qwen` — local `Qwen/Qwen3-Embedding-0.6B` via
  `sentence_transformers`. No external embedding API calls after model
  download. Drift vs published is small for relative ranking; absolute scores
  on the 9 embedding-driven benches will not match exactly.

The lower-level metric factory defaults to `gemini` for release-set fidelity.
`run_with_agc_judge.sh` overrides that default to `qwen` so users without
paid Google billing can complete a full eval; set
`AGC_EMBEDDING_BACKEND=gemini` when reproducing exact published-cohort numbers.

### Hypobench Inference

One bench, `hypobench`, runs a downstream classifier on ~800 test rows per
instance (predict a label given a generated hypothesis + a row of test data).
By default this classifier call uses `google/gemini-3-flash-preview` and is not routed through `AGC_JUDGE_OVERRIDE`. Override with `HYPOBENCH_INFERENCE_MODEL_OVERRIDE=<model>` if you want a different classifier backend. For a smoke test, you can also cap the classifier rows with `HYPOBENCH_MAX_IND_EXAMPLES` and `HYPOBENCH_MAX_OOD_EXAMPLES`; leave those unset for a full run.

---

## Setup

### 1. Obtain the bundle

Use this repository checkout, or download the matching snapshot from [huggingface.co/datasets/agcbench-2026/AGC-Bench](https://huggingface.co/datasets/agcbench-2026/AGC-Bench). The AGC-Judge model is released separately at [huggingface.co/agcbench-2026/AGC-Judge](https://huggingface.co/agcbench-2026/AGC-Judge). From either source, `cd` into the bundle root before running the commands below.

### 2. Create a Python 3.10 environment

```bash
conda create --name agc-env python=3.10 -y
conda activate agc-env
```

### 3. Install the package

```bash
pip install -e ".[eval,dev]"
```

Pulls `crfm-helm>=0.5.12`, data-format libraries (Pillow, h5py, openpyxl, PyYAML), optional eval dependencies (diffusers, clip-score), and pytest.

### 4. Configure API keys for live model evaluation

The reproduction scripts above do not require API keys. Live model evaluation requires credentials for the model under evaluation and for any external services used by the selected scoring path.

Most target-model calls route through OpenRouter, which provides unified access to Claude, GPT, Llama, and other systems through a single credential. Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

If you use `AGC_EMBEDDING_BACKEND=gemini`, or if you run source-paper judge configurations that call Google's Gemini API directly, set a Google API key as well:

```bash
cat > .env << 'EOF'
export OPENROUTER_API_KEY="sk-or-..."
export GOOGLE_API_KEY="..."   # required for google/* judges (poetmt + JRT calibration)
EOF
source .env
```

For the AGC-Judge HF Endpoint path, also copy `prod_env/credentials.conf.template` to `prod_env/credentials.conf` and fill `huggingfaceApiToken`.

### 5. Sanity check

```bash
python -c "from helm.benchmark.run import main; print('helm OK')"
python -c "import pathlib; n = len(list(pathlib.Path('scenarios').glob('*_scenario.py'))); print(f'scenarios OK ({n} found)')"
```

---

## Advanced: Raw HELM Orchestrator

For new-model leaderboard scoring, prefer [eval_scripts/run_with_agc_judge.sh](eval_scripts/run_with_agc_judge.sh), which routes the 24 LLM-judge benchmarks through AGC-Judge and then integrates the new model into the released leaderboard scale.

[eval_scripts/00_run_all_parallel.sh](eval_scripts/00_run_all_parallel.sh) is the lower-level HELM orchestrator. It verifies the target model and, when `AGC_JUDGE_OVERRIDE` is not set, the source judge models referenced by the run specs. It then runs dataset scripts concurrently. By default it restricts evaluation to the 67 text-only primary datasets listed as `included` in `release_data/dataset_metadata.csv`; set `AGC_INCLUDE_MULTIMODAL=1` to evaluate all 78 scenarios on disk.

```bash
bash eval_scripts/00_run_all_parallel.sh MODEL [MAX_INSTANCES] [PARALLELISM]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `MODEL` | yes | — | OpenRouter model identifier (`vendor/model`) |
| `MAX_INSTANCES` | no | `-1` (all) | Cap on instances per dataset (use a small number for smoke tests) |
| `PARALLELISM` | no | `4` | Number of datasets to run concurrently |

Examples:

```bash
# Primary text-only evaluation, 4-way parallel
bash eval_scripts/00_run_all_parallel.sh google/gemini-2.5-flash-lite

# Smoke test: 10 instances per dataset, 8-way parallel
bash eval_scripts/00_run_all_parallel.sh openai/gpt-4o-mini 10 8

# Primary text-only evaluation, higher concurrency
bash eval_scripts/00_run_all_parallel.sh openai/gpt-4o -1 16
```

The orchestrator loads `OPENROUTER_API_KEY` from `.env`, verifies the target model and required judge identifiers are reachable, then dispatches each dataset's `eval_scripts/<dataset>.sh` in the background, throttled to `PARALLELISM` workers. HELM downloads any missing dataset data on first run. Per-dataset logs land at `benchmark_output/runs/first_full_trial/_orchestrator_logs/<dataset>.log`; final results at `benchmark_output/runs/first_full_trial/<run_dir>/`. The orchestrator exits non-zero if any dataset failed.

| Exit | Meaning |
|---|---|
| 0 | All datasets passed |
| 1 | One or more datasets failed or were skipped |
| 2 | Bad arguments, missing `OPENROUTER_API_KEY`, or OpenRouter list fetch failed |
| 3 | Target `MODEL` not available on OpenRouter |
| 4 | A required judge model not available on OpenRouter |
| 5 | Dataset list file missing |
| 6 | Dataset list empty |

To debug or rerun one dataset:

```bash
bash eval_scripts/<dataset>.sh "$MODEL" first_full_trial ""
```

The third argument is `MAX_INSTANCES` (empty = all).

---

## License

Dual-licensed:

- **Code** (the harness, scoring code, judge prompts, analysis scripts) — Apache License 2.0; see [LICENSE](LICENSE).
- **Data** (`release_data/`, `analysis/`, `croissant/`, [DATASHEET.md](DATASHEET.md), `data/registry/`) — Creative Commons Attribution 4.0 International (CC BY 4.0); see [LICENSE-DATA](LICENSE-DATA).

Per-benchmark source data retains its source-paper license, documented in the released registry and per-benchmark Croissant manifests.
