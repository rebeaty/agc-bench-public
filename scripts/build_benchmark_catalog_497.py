#!/usr/bin/env python3
"""Build the 497-row PRISMA benchmark catalog.

This script joins catalog-stage files into one release table:

  extracted benchmark mentions -> deduplicated 497 benchmark records
  -> 432 creativity-relevant candidates -> 78 onboarded release artifacts

The script starts at the benchmark-catalog stage, where the checked-in files
reproduce the paper's 497 and 432 counts exactly. Earlier paper-harvest and
screening counts are documented in the paper; those transient logs are not
treated as canonical release files here. By default, the script rebuilds from
curation/catalog/source; pass --archive to rebuild from a compatible source
archive with the original directory layout.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO / "curation/catalog/source"


def norm(value: object) -> str:
    """Normalize benchmark names for catalog joins."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def split_variants(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def first_nonempty(values: list[object]) -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return ""


def join_unique(values: list[object], limit: int | None = None) -> str:
    seen: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    if limit is not None:
        seen = seen[:limit]
    return " | ".join(seen)


def load_extraction_index(path: Path) -> dict[str, dict]:
    """Return normalized benchmark-name -> extraction summary."""
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        paper = json.loads(line)
        for bench in paper.get("benchmarks", []) or []:
            name = bench.get("name") or bench.get("benchmark_name")
            key = norm(name)
            if not key:
                continue
            tasks = bench.get("tasks", []) or []
            grouped[key]["benchmark_name"].append(name)
            grouped[key]["paper_id"].append(paper.get("paper_id"))
            grouped[key]["paper_title"].append(paper.get("title"))
            grouped[key]["paper_year"].append(paper.get("year"))
            grouped[key]["paper_doi"].append(paper.get("doi"))
            grouped[key]["benchmark_relevance"].append(bench.get("relevance"))
            grouped[key]["benchmark_status"].append(bench.get("status"))
            grouped[key]["task_count"].append(len(tasks))
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                grouped[key]["task_name"].append(task.get("name"))
                grouped[key]["task_type"].append(task.get("task_type"))
                grouped[key]["modality"].append(task.get("modality"))
                grouped[key]["scoring_method"].append(task.get("scoring_method"))
                grouped[key]["scoring_detail"].append(task.get("scoring_detail"))

    out: dict[str, dict] = {}
    for key, values in grouped.items():
        out[key] = {
            "extraction_benchmark_name": first_nonempty(values["benchmark_name"]),
            "extraction_paper_id": first_nonempty(values["paper_id"]),
            "extraction_paper_title": first_nonempty(values["paper_title"]),
            "extraction_paper_year": first_nonempty(values["paper_year"]),
            "extraction_paper_doi": first_nonempty(values["paper_doi"]),
            "extraction_benchmark_relevance": join_unique(values["benchmark_relevance"]),
            "extraction_benchmark_status": join_unique(values["benchmark_status"]),
            "extraction_task_rows": sum(int(v or 0) for v in values["task_count"]),
            "task_types": join_unique(values["task_type"]),
            "extraction_modalities": join_unique(values["modality"]),
            "scoring_method": join_unique(values["scoring_method"]),
            "scoring_detail": join_unique(values["scoring_detail"], limit=8),
        }
    return out


def source_paths(root: Path) -> dict[str, Path]:
    """Return source-file paths for either the release or archive layout."""
    release_layout = {
        "extraction": root / "extracted_benchmarks_merged.jsonl",
        "unique": root / "unique_benchmarks.csv",
        "dedup": root / "deduplication_results.json",
        "relevance": root / "benchmark_catalog_432.csv",
    }
    if release_layout["unique"].exists():
        return release_layout

    return {
        "extraction": root / "extraction_output/extracted_benchmarks_merged.jsonl",
        "unique": root / "deduplication_output/unique_benchmarks.csv",
        "dedup": root / "deduplication_output/deduplication_results.json",
        "relevance": root / "benchmark_catalog/benchmark_catalog.csv",
    }


def build_catalog(source_root: Path) -> pd.DataFrame:
    paths = source_paths(source_root)
    unique = pd.read_csv(paths["unique"])
    relevance = pd.read_csv(paths["relevance"])
    extraction = load_extraction_index(paths["extraction"])
    extraction_by_paper: dict[str, list[dict]] = defaultdict(list)
    for summary in extraction.values():
        paper_id = summary.get("extraction_paper_id")
        if paper_id:
            extraction_by_paper[str(paper_id)].append(summary)

    relevance_by_name = {norm(row.benchmark_name): row for row in relevance.itertuples(index=False)}

    rows: list[dict] = []
    for record in unique.itertuples(index=False):
        names = [record.canonical_name] + split_variants(record.variant_names)

        relevance_match = None
        relevance_match_name = ""
        for name in names:
            hit = relevance_by_name.get(norm(name))
            if hit is not None:
                relevance_match = hit
                relevance_match_name = name
                break

        extraction_candidates: list[tuple[str, dict]] = []
        for name in names:
            hit = extraction.get(norm(name))
            if hit is not None:
                extraction_candidates.append((name, hit))
        extraction_match_name = ""
        extraction_match = None
        if extraction_candidates:
            # Some dedup variants point to an alias row with no task payload
            # while another alias for the same benchmark has the useful
            # scoring/task extraction. Prefer the richer record.
            extraction_match_name, extraction_match = max(
                extraction_candidates,
                key=lambda item: (
                    bool(item[1].get("scoring_method")),
                    int(item[1].get("extraction_task_rows") or 0),
                ),
            )
        if extraction_match is None or not extraction_match.get("scoring_method"):
            same_paper_scored = [
                item
                for item in extraction_by_paper.get(str(record.origin_paper_id), [])
                if item.get("scoring_method")
            ]
            if len(same_paper_scored) == 1:
                extraction_match = same_paper_scored[0]
                extraction_match_name = extraction_match.get("extraction_benchmark_name", "")

        row = {
            "benchmark_name": record.canonical_name,
            "aliases": record.variant_names,
            "creativity_relevant": bool(relevance_match is not None),
            "source_paper_title": record.origin_title,
            "source_paper_year": record.origin_year,
            "source_paper_doi": record.origin_doi,
            "source_paper_semantic_scholar_id": record.origin_paper_id,
            "modality": "",
            "runnability_tier": "",
            "scoring_method": "",
            "scoring_protocol": "",
            "task_types": "",
        }

        if relevance_match is not None:
            row.update({
                "modality": relevance_match.modality,
                "runnability_tier": relevance_match.tier_name,
            })

        if extraction_match is not None:
            row["scoring_method"] = extraction_match.get("scoring_method", "")
            row["scoring_protocol"] = extraction_match.get("scoring_detail", "")
            row["task_types"] = extraction_match.get("task_types", "")

        rows.append(row)

    return pd.DataFrame(rows)


def copy_source_files(source_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = source_paths(source_root)
    copies = {
        paths["extraction"]: out_dir / "extracted_benchmarks_merged.jsonl",
        paths["unique"]: out_dir / "unique_benchmarks.csv",
        paths["dedup"]: out_dir / "deduplication_results.json",
        paths["relevance"]: out_dir / "benchmark_catalog_432.csv",
    }
    for src, dst in copies.items():
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=(
            "Optional compatible source archive root. Defaults to the checked-in "
            "curation/catalog/source files."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "release_data/benchmark_catalog_497.csv",
    )
    parser.add_argument(
        "--source-out",
        type=Path,
        default=REPO / "curation/catalog/source",
    )
    args = parser.parse_args()

    source_root = args.archive or DEFAULT_SOURCE
    catalog = build_catalog(source_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.out, index=False)
    copy_source_files(source_root, args.source_out)

    n_rows = len(catalog)
    n_relevant = int(catalog["creativity_relevant"].sum())
    n_filtered = n_rows - n_relevant
    n_scoring = int((catalog["scoring_method"].astype(str).str.len() > 0).sum())
    print(f"Wrote {args.out}")
    print(f"Rows: {n_rows}")
    print(f"Creativity-relevant: {n_relevant}")
    print(f"Filtered adjacent/general: {n_filtered}")
    print(f"Rows with extracted scoring protocol: {n_scoring}")
    print(f"Copied source files to {args.source_out}")

    if n_rows != 497 or n_relevant != 432 or n_filtered != 65:
        raise SystemExit("Unexpected catalog counts; refusing to silently continue.")


if __name__ == "__main__":
    main()
