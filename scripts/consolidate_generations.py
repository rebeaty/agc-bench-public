#!/usr/bin/env python3
"""Build the release-subset generation corpus from retained HELM run outputs.

For each cell in release_data/long_model_x_dataset.csv (5,510 release-set
cells across 83 models x 67 datasets):

  1. Find the canonical run dir under benchmark_output/runs/.
  2. Read scenario_state.json (prompts + completions per instance).
  3. Join per_instance_stats.json after filtering to the released canonical
     metric for that benchmark.
  4. Join agc_judge_per_item.csv where applicable (LLM-judge cells).
  5. Write generations/<provider>/<model_slug>/<benchmark>.parquet with
     completions, released cell-score metadata, and per-instance canonical
     scores when HELM exposed them.
  6. Write de-duplicated prompts to generations/prompts/<benchmark>.parquet
     unless --include-prompts is set.
  7. Log per-cell audit row to generations/_consolidation_audit.csv.

Run modes:
  --dry-run N      Process only N cells (default: 3), output to /tmp/agc-gen-dryrun/
  --full           Process all cells, output to ./generations/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- Paths ----------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get(
    "AGC_RUNS_DIR",
    str(REPO / "benchmark_output" / "runs"),
))
LONG_CSV = REPO / "release_data" / "long_model_x_dataset.csv"
JUDGE_CSV = REPO / "release_data" / "agc_judge_per_item.csv"
CANONICAL_CSV = REPO / "release_data" / "dataset_raw_distribution.csv"

# --- Run-directory priority and skip rules -------------------------------
# Some cells have multiple retained runs from targeted reruns and
# token-limit checks.
# Directory prefixes are ordered from most authoritative to least authoritative
# for the frozen release set. Within a tier, the newest populated run wins.
RUN_PRIORITY = [
    "oai_backfill_v1",
    "truncation_patch_v2",
    "truncation_patch_v1",
    "modal_phase1c_n50_mmext",
    "modal_phase1c_n50",
    "openrouter_text_promoted_scale_v1_phase1c",
    "openrouter_mm_promoted_scale_v1_phase1c",
    "openrouter_text_promoted_scale_v1_phase1b_n20",
    "openrouter_mm_promoted_scale_v1_phase1b",
    "openrouter_text_promoted_scale_v1_phase1_n5",
    "openrouter_text_promoted_scale_v1",
    "openrouter_mm_promoted_scale_v1",
    "openrouter_text_admission_scale_v1",
    "openrouter_mm_admission_reset_v1",
    "big_g_2_text_v1",
    "big_g_2_mm_v1",
    "big_g_google_launch_v3",
    "big_g_google_launch_v2",
    "topup_text_v1_or",
    "topup_text_v1_direct",
    "topup_text_v1",
    "topup_mm_v1",
]

SKIP_PATTERNS = [
    re.compile(r"^_archive"),
    re.compile(r"_smoke"),
    re.compile(r"^probe_"),
    re.compile(r"_pilot_"),
    re.compile(r"^cachefix"),
    re.compile(r"^_truncation_patch_archive"),
    re.compile(r"^fix_pass_"),
    re.compile(r":model=.*:thinking$"),  # thinking-mode variants
]

# --- Helpers -------------------------------------------------------------
def model_to_cell_slug(model: str) -> str:
    """`openai/gpt-5.4-mini` -> `openai_gpt-5.4-mini`. Only / replaced."""
    return model.replace("/", "_")


def list_candidate_run_dirs(benchmark: str, model: str, runs_dir: Path = RUNS_DIR) -> list[Path]:
    """All run dirs that contain a non-empty <benchmark>:model=<slug> cell."""
    cell_slug = model_to_cell_slug(model)
    cell_name = f"{benchmark}:model={cell_slug}"
    matches = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if any(p.search(run_name) for p in SKIP_PATTERNS):
            continue
        cell_dir = run_dir / cell_name
        if not cell_dir.exists():
            continue
        # Non-empty if scenario_state.json exists and is > 100 bytes
        ss = cell_dir / "scenario_state.json"
        if ss.exists() and ss.stat().st_size > 100:
            matches.append(cell_dir)
    return matches


def pick_canonical(candidates: list[Path]) -> Path | None:
    """Select the release-preferred run directory for one cell."""
    if not candidates:
        return None
    def priority(p: Path) -> tuple[int, float]:
        run_name = p.parent.name
        for i, prefix in enumerate(RUN_PRIORITY):
            if run_name.startswith(prefix):
                return (i, -p.stat().st_mtime)
        return (len(RUN_PRIORITY), -p.stat().st_mtime)
    return min(candidates, key=priority)


def stat_name(stat: dict) -> str | None:
    name = stat.get("name")
    if isinstance(name, dict):
        return name.get("name")
    return name


def load_cell(cell_dir: Path, canonical_metric: str) -> pd.DataFrame | None:
    """Read scenario_state.json + per_instance_stats.json, return joined df.
    Returns None if the cell can't be read."""
    ss_path = cell_dir / "scenario_state.json"
    pis_path = cell_dir / "per_instance_stats.json"

    try:
        ss = json.loads(ss_path.read_text())
    except Exception as e:
        print(f"  ! could not read {ss_path}: {e}", file=sys.stderr)
        return None

    rows = []
    for rs in ss.get("request_states", []):
        inst = rs.get("instance", {}) or {}
        inp = inst.get("input", {}) or {}
        prompt_text = ""
        req = rs.get("request", {}) or {}
        if isinstance(req, dict):
            prompt_text = req.get("prompt", "") or ""
        if not prompt_text and isinstance(inp, dict):
            prompt_text = inp.get("text", "") or ""

        completion = ""
        result = rs.get("result")
        if result and isinstance(result, dict):
            comps = result.get("completions") or []
            if comps and isinstance(comps[0], dict):
                completion = comps[0].get("text", "") or ""

        rows.append({
            "instance_id": inst.get("id"),
            "prompt": prompt_text,
            "completion": completion,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Join released canonical scores from per_instance_stats.json. HELM writes
    # many bookkeeping and auxiliary metrics per instance; keeping all of them
    # expands one generation into multiple rows and makes per-cell means
    # meaningless. The release_data/dataset_raw_distribution.csv whitelist is
    # the metric that best reproduces the published dataset_z for each dataset.
    df["canonical_metric"] = canonical_metric
    df["canonical_score"] = pd.NA
    if pis_path.exists() and pis_path.stat().st_size > 10:
        try:
            pis = json.loads(pis_path.read_text())
            score_rows = []
            for entry in pis:
                inst_id = entry.get("instance_id") or (
                    entry.get("instance") or {}).get("id")
                stats = entry.get("stats") or []
                for stat in stats:
                    name = stat_name(stat)
                    if name != canonical_metric:
                        continue
                    score = stat.get("sum")
                    score_rows.append({
                        "instance_id": inst_id,
                        "canonical_metric": name,
                        "canonical_score": score,
                    })
            if score_rows:
                scores_df = pd.DataFrame(score_rows)
                scores_df["canonical_score"] = pd.to_numeric(
                    scores_df["canonical_score"], errors="coerce"
                )
                scores_df = (
                    scores_df
                    .groupby(["instance_id", "canonical_metric"], as_index=False)
                    .agg(canonical_score=("canonical_score", "mean"))
                )
                df = df.drop(columns=["canonical_metric", "canonical_score"])
                df = df.merge(scores_df, on="instance_id", how="left")
        except Exception as e:
            print(f"  ! could not parse {pis_path}: {e}", file=sys.stderr)

    return df


def join_judge_scores(df: pd.DataFrame, model: str, benchmark: str,
                       judge_df: pd.DataFrame) -> pd.DataFrame:
    """Add agc_judge_score and jrt_gold for LLM-judge cells."""
    sub = judge_df[(judge_df["model"] == model) &
                   (judge_df["benchmark"] == benchmark)]
    if sub.empty:
        df["agc_judge_score"] = pd.NA
        df["jrt_gold"] = pd.NA
        return df
    return df.merge(
        sub[["item_id", "agc_judge_score", "jrt_gold"]].rename(
            columns={"item_id": "instance_id"}),
        on="instance_id", how="left"
    )


# --- Main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=0,
                    help="Process only first N cells (output to /tmp/agc-gen-dryrun/)")
    ap.add_argument("--full", action="store_true",
                    help="Process all cells (output to ./generations/)")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                    help="HELM benchmark_output/runs directory")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Output directory (default: /tmp/agc-gen-dryrun or ./generations)")
    ap.add_argument("--datasets", default="",
                    help="Optional comma-separated dataset filter")
    ap.add_argument("--models", default="",
                    help="Optional comma-separated model filter")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional post-filter cell limit")
    ap.add_argument("--include-prompts", action="store_true",
                    help="Keep prompt text in every per-cell parquet instead of writing a de-duplicated prompt bank")
    ap.add_argument("--prompt-root", type=Path, default=None,
                    help="Prompt-bank output directory (default: <out-root>/prompts)")
    args = ap.parse_args()

    if not args.dry_run and not args.full:
        ap.error("specify --dry-run N or --full")

    out_root = args.out_root or (Path("/tmp/agc-gen-dryrun") if args.dry_run else Path("generations"))
    out_root.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(LONG_CSV)
    judge_df = pd.read_csv(JUDGE_CSV)
    canonical_df = pd.read_csv(CANONICAL_CSV)
    canonical_metric = dict(zip(canonical_df["dataset"], canonical_df["canonical_metric"]))
    cell_metadata = long_df.set_index(["model", "dataset"]).to_dict("index")

    cells = long_df[["model", "dataset"]].drop_duplicates().to_dict("records")
    dataset_filter = {x.strip() for x in args.datasets.split(",") if x.strip()}
    model_filter = {x.strip() for x in args.models.split(",") if x.strip()}
    if dataset_filter:
        cells = [c for c in cells if c["dataset"] in dataset_filter]
    if model_filter:
        cells = [c for c in cells if c["model"] in model_filter]
    if args.dry_run:
        # Pick a diverse sample: one popular text, one LLM-judge, one multimodal
        sample = []
        seen_models = set()
        for c in cells:
            if len(sample) >= args.dry_run:
                break
            if c["model"] in seen_models:
                continue
            sample.append(c)
            seen_models.add(c["model"])
        cells = sample
    if args.limit:
        cells = cells[:args.limit]

    audit_rows = []
    prompt_bank: dict[str, dict[str, str]] = defaultdict(dict)
    prompt_conflicts: dict[str, int] = defaultdict(int)
    print(f"Processing {len(cells)} cells -> {out_root}")
    for i, cell in enumerate(cells, 1):
        model, benchmark = cell["model"], cell["dataset"]
        print(f"\n[{i}/{len(cells)}] {model} x {benchmark}")
        metric = canonical_metric.get(benchmark)
        if not metric:
            print("  - SKIP: no released canonical metric found")
            audit_rows.append({
                "model": model, "benchmark": benchmark,
                "canonical_metric": "",
                "status": "skipped_no_canonical_metric", "n_candidates": 0,
                "source_run_dir": "", "n_instances": 0, "n_rows": 0,
                "n_scored_instances": 0,
            })
            continue
        candidates = list_candidate_run_dirs(benchmark, model, args.runs_dir)
        canonical = pick_canonical(candidates)
        if canonical is None:
            print(f"  - SKIP: no populated run dir found ({len(candidates)} cands)")
            audit_rows.append({
                "model": model, "benchmark": benchmark,
                "canonical_metric": metric,
                "status": "skipped_no_data", "n_candidates": 0,
                "source_run_dir": "", "n_instances": 0, "n_rows": 0,
                "n_scored_instances": 0,
            })
            continue

        df = load_cell(canonical, metric)
        if df is None or df.empty:
            print(f"  - SKIP: cell unreadable {canonical}")
            audit_rows.append({
                "model": model, "benchmark": benchmark,
                "canonical_metric": metric,
                "status": "skipped_unreadable",
                "n_candidates": len(candidates),
                "source_run_dir": canonical.parent.name, "n_instances": 0,
                "n_rows": 0, "n_scored_instances": 0,
            })
            continue

        df = join_judge_scores(df, model, benchmark, judge_df)
        df["model"] = model
        df["benchmark"] = benchmark
        df["source_run_dir"] = canonical.parent.name
        meta = cell_metadata.get((model, benchmark), {})
        df["dataset_z"] = meta.get("dataset_z", pd.NA)
        df["score_source"] = meta.get("score_source", pd.NA)
        df["n_score_sources"] = meta.get("n_score_sources", pd.NA)
        df["dq_masked"] = meta.get("dq_masked", pd.NA)

        write_df = df
        if not args.include_prompts and "prompt" in df.columns:
            for item in df[["instance_id", "prompt"]].dropna().itertuples(index=False):
                inst_id = str(item.instance_id)
                prompt = str(item.prompt)
                existing = prompt_bank[benchmark].get(inst_id)
                if existing is None:
                    prompt_bank[benchmark][inst_id] = prompt
                elif existing != prompt:
                    prompt_conflicts[benchmark] += 1
            write_df = df.drop(columns=["prompt"])

        provider = model.split("/")[0]
        model_slug = model.split("/", 1)[1].replace("/", "_")
        out_dir = out_root / provider / model_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{benchmark}.parquet"
        write_df.to_parquet(out_path, index=False, compression="zstd")
        n_instances = int(df["instance_id"].nunique(dropna=True))
        n_scored = int(df["canonical_score"].notna().sum())
        print(f"  -> wrote {len(df)} rows / {n_instances} instances to {out_path}")

        audit_rows.append({
            "model": model, "benchmark": benchmark,
            "canonical_metric": metric,
            "status": "ok", "n_candidates": len(candidates),
            "source_run_dir": canonical.parent.name,
            "n_instances": n_instances,
            "n_rows": len(df),
            "n_scored_instances": n_scored,
        })

    audit_path = out_root / "_consolidation_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    if not args.include_prompts:
        prompt_root = args.prompt_root or (out_root / "prompts")
        prompt_root.mkdir(parents=True, exist_ok=True)
        prompt_audit_rows = []
        for benchmark, prompts in sorted(prompt_bank.items()):
            prompt_df = pd.DataFrame(
                [
                    {"benchmark": benchmark, "instance_id": inst_id, "prompt": prompt}
                    for inst_id, prompt in sorted(prompts.items())
                ]
            )
            prompt_df.to_parquet(
                prompt_root / f"{benchmark}.parquet",
                index=False,
                compression="zstd",
            )
            prompt_audit_rows.append({
                "benchmark": benchmark,
                "n_prompts": len(prompt_df),
                "n_prompt_conflicts": prompt_conflicts.get(benchmark, 0),
            })
        pd.DataFrame(prompt_audit_rows).to_csv(
            prompt_root / "_prompt_audit.csv", index=False
        )
    print(f"\nAudit log -> {audit_path}")
    if not args.include_prompts:
        print(f"Prompt bank -> {prompt_root}")
    print(f"Done.  ok={sum(1 for r in audit_rows if r['status']=='ok')}  "
          f"skipped={sum(1 for r in audit_rows if r['status']!='ok')}")


if __name__ == "__main__":
    main()
