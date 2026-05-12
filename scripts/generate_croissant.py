"""Generate Croissant JSON-LD metadata for AGC-Bench.

Produces:
  croissant/agc_bench.json              -- aggregate manifest for the
                                            full release (catalog +
                                            harness + outputs +
                                            validation analyses)
  croissant/per_benchmark/<benchmark_id>.json -- one manifest per onboarded
                                            benchmark, citing its
                                            source paper

The aggregate manifest describes the full release package. Per-benchmark
manifests document the source paper, task, metric, and release artifacts for
each onboarded benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _domain_mapping import DOMAIN_MAPPING

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / 'croissant'
PER_BENCH_DIR = OUT_DIR / 'per_benchmark'

DATASET_URL = 'https://huggingface.co/datasets/agcbench-2026/AGC-Bench'
MODEL_URL = 'https://huggingface.co/agcbench-2026/AGC-Judge'
PROJECT_URL = DATASET_URL  # canonical homepage for the dataset artifact
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/'
RELEASE_VERSION = '1.0.0'
AGGREGATE_METADATA_VERSION = '1.0.1'
CROISSANT_CONFORMS_TO = 'http://mlcommons.org/croissant/1.1'

# Bundle files referenced as Croissant FileObjects. Paths are relative to the
# dataset repo root (the same paths exist locally in this checkout). SHA256
# hashes are computed at manifest-generation time over the actual file bytes.
DISTRIBUTION_FILES = [
    {
        '@id': 'agc-bench-catalog',
        'name': 'agc-bench-catalog.csv',
        'local_path': 'croissant/agc-bench-catalog.csv',
        'description': (
            'Per-benchmark catalog of the 78 onboarded benchmarks (67 text-'
            'only, 11 multimodal) plus 14 excluded candidates. Includes '
            'status, domain, JRT-correction flag, source-paper canonical '
            'metric, and exclusion rationale where applicable.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-catalog-497',
        'name': 'benchmark_catalog_497.csv',
        'local_path': 'release_data/benchmark_catalog_497.csv',
        'description': (
            'v1.0.1 catalog with one row per deduplicated PRISMA benchmark '
            'record. The creativity_relevant flag marks the 432 creativity-'
            'relevant onboarding candidates and retains the 65 adjacent/'
            'general extracted benchmarks for auditability.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-catalog-audit-readme',
        'name': 'catalog-audit-readme.md',
        'local_path': 'curation/catalog/README.md',
        'description': (
            'Catalog note explaining the 497 deduplicated records, the 432 '
            'creativity-relevant subset, and the catalog-stage selection record.'
        ),
        'encodingFormat': 'text/markdown',
    },
    {
        '@id': 'agc-bench-extracted-benchmarks',
        'name': 'extracted_benchmarks_merged.jsonl',
        'local_path': 'curation/catalog/source/extracted_benchmarks_merged.jsonl',
        'description': (
            'Catalog-stage extraction output: 283 papers, 546 benchmark '
            'mentions, and 1,160 task records before deduplication.'
        ),
        'encodingFormat': 'application/jsonl',
    },
    {
        '@id': 'agc-bench-unique-benchmarks',
        'name': 'unique_benchmarks.csv',
        'local_path': 'curation/catalog/source/unique_benchmarks.csv',
        'description': (
            'Deduplicated catalog-stage benchmark records: 497 unique records '
            'after hybrid string and consensus deduplication.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-deduplication-results',
        'name': 'deduplication_results.json',
        'local_path': 'curation/catalog/source/deduplication_results.json',
        'description': (
            'Deduplication grouping metadata supporting the 497-row benchmark '
            'catalog.'
        ),
        'encodingFormat': 'application/json',
    },
    {
        '@id': 'agc-bench-creativity-relevant-catalog',
        'name': 'benchmark_catalog_432.csv',
        'local_path': 'curation/catalog/source/benchmark_catalog_432.csv',
        'description': (
            'Creativity-relevance filtered catalog: 432 benchmark candidates '
            'that remained after removing adjacent/general co-extractions.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-onboarder-readme',
        'name': 'onboarder-readme.md',
        'local_path': 'curation/onboarder/README.md',
        'description': (
            'README for the benchmark onboarding workflow and its supporting '
            'scenario-implementation records.'
        ),
        'encodingFormat': 'text/markdown',
    },
    {
        '@id': 'agc-bench-onboarding-instructions',
        'name': 'SKILL.md',
        'local_path': 'curation/onboarder/SKILL.md',
        'description': (
            'Operational benchmark-onboarding instructions used to convert '
            'catalog candidates into HELM-compatible scenario implementations.'
        ),
        'encodingFormat': 'text/markdown',
    },
    {
        '@id': 'agc-bench-onboarder-queue',
        'name': 'benchmarks.json',
        'local_path': 'curation/onboarder/benchmarks.json',
        'description': (
            'Anonymized benchmark-onboarding queue and status metadata used '
            'during scenario curation.'
        ),
        'encodingFormat': 'application/json',
    },
    {
        '@id': 'agc-bench-onboarder-template',
        'name': 'helm-template.md',
        'local_path': 'curation/onboarder/helm-template.md',
        'description': 'HELM scenario template and implementation patterns for onboarding.',
        'encodingFormat': 'text/markdown',
    },
    {
        '@id': 'agc-bench-onboarder-example-multimodal',
        'name': 'multimodal_visual_qa.py',
        'local_path': 'curation/onboarder/examples/multimodal_visual_qa.py',
        'description': (
            'Example HELM Scenario pattern for benchmarks with image inputs.'
        ),
        'encodingFormat': 'text/x-python',
    },
    {
        '@id': 'agc-bench-leaderboard',
        'name': 'agc-bench-leaderboard.csv',
        'local_path': 'release_data/leaderboard.csv',
        'description': (
            'Per-release-model leaderboard with mean / median z, dataset coverage '
            'count, and rank, computed under JRT-corrected scoring with the '
            'data-quality mask applied.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-viewer-scores',
        'name': 'model_dataset_scores.csv',
        'local_path': 'release_data/model_dataset_scores.csv',
        'description': (
            'Compact per-(model, dataset) z-score table used as the default '
            'Hugging Face Dataset Viewer surface. Includes model, dataset, '
            'dataset_z, score_source, and dq_masked.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-long-scores',
        'name': 'agc-bench-long-scores.csv',
        'local_path': 'release_data/long_model_x_dataset.csv',
        'description': (
            'Per-(model, dataset) z-score table with score-source flag '
            '(jrt vs. raw), aggregate score-source count, and a dq_masked '
            'boolean for cells removed by the data-quality sweep.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-dataset-metadata',
        'name': 'dataset_metadata.csv',
        'local_path': 'release_data/dataset_metadata.csv',
        'description': (
            'Per-dataset release metadata with status, domain, release-set '
            'model coverage count, canonical-metric count, JRT-correction flag, '
            'and exclusion notes where applicable.'
        ),
        'encodingFormat': 'text/csv',
    },
    {
        '@id': 'agc-bench-cfactor',
        'name': 'agc-bench-cfactor.csv',
        'local_path': 'analysis/c_factor_loadings.csv',
        'description': (
            'Per-domain c-factor loadings, eigenvalues, and robustness-check '
            'outputs. Companion to Section 4.3 of the paper.'
        ),
        'encodingFormat': 'text/csv',
    },
]

DISTRIBUTION_FILESETS = [
    {
        '@id': 'agc-bench-generation-cells',
        'name': 'generation-cell-parquets',
        'includes': 'generations/*/*/*.parquet',
        'description': (
            'Derived per-(model, dataset) generation corpus: one parquet per '
            'release cell with instance_id, completion, released cell-score '
            'metadata, per-instance canonical score where HELM exposed it, '
            'optional AGC-Judge fields, model, dataset, and source run '
            'directory.'
        ),
        'encodingFormat': 'application/x-parquet',
    },
    {
        '@id': 'agc-bench-generation-prompts',
        'name': 'generation-prompt-bank',
        'includes': 'generations/prompts/*.parquet',
        'description': (
            'De-duplicated prompt bank keyed by benchmark and instance_id for '
            'the generation-cell parquet corpus.'
        ),
        'encodingFormat': 'application/x-parquet',
    },
]


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def _file_object(spec):
    """Build a Croissant FileObject from a DISTRIBUTION_FILES spec."""
    local = REPO / spec['local_path']
    sha = _sha256(local) if local.exists() else 'missing-local-file'
    size = local.stat().st_size if local.exists() else None
    return {
        '@type': 'cr:FileObject',
        '@id': spec['@id'],
        'name': spec['name'],
        'description': spec['description'],
        'contentUrl': f"{DATASET_URL}/resolve/main/{spec['local_path']}",
        'encodingFormat': spec['encodingFormat'],
        'sha256': sha,
        **({'contentSize': str(size)} if size is not None else {}),
    }


def _file_set(spec):
    """Build a Croissant FileSet for derived multi-file artifacts."""
    return {
        '@type': 'cr:FileSet',
        '@id': spec['@id'],
        'name': spec['name'],
        'description': spec['description'],
        'contentUrl': f"{DATASET_URL}/tree/main/{spec['includes'].split('/')[0]}",
        'encodingFormat': spec['encodingFormat'],
        'includes': spec['includes'],
    }


def _aggregate_source_datasets():
    """Return source links for the aggregate manifest as a sorted union
    of source-paper and source-repo links across all released benchmarks."""
    reg = yaml.safe_load((REPO / 'data/registry/registry_master.yaml').read_text())['datasets']
    uris = set()
    for k, v in reg.items():
        if not v.get('released_in_v1'):
            continue
        for field in ('source_paper', 'source_repo'):
            u = v.get(field)
            if u and isinstance(u, str) and u.startswith('http'):
                uris.add(u)
    return sorted(uris)


def _source_dataset_entries(uris):
    """Format source links for the RAI source field."""
    return [{"@id": u} for u in uris if u]


def _agent(label: str, *, software: bool = False, description: str | None = None):
    """Build a compact PROV agent entry for RAI activities."""
    out = {
        "@type": "prov:SoftwareAgent" if software else "prov:Agent",
        "prov:label": label,
    }
    if description:
        out["sc:description"] = description
    return out


ACTIVITY_TYPE_URIS = {
    "Data Collection": "https://www.wikidata.org/wiki/Q4929239",
    "Annotation": "https://www.wikidata.org/wiki/Q109719325",
    "Data preprocessing": "https://www.wikidata.org/wiki/Q5227332",
    "Quality review": "https://www.wikidata.org/wiki/Q3306762",
}


def _prov_activity(label: str, activity_type: str, description: str, agents=None):
    """Build a PROV activity using the schema consumed by the RAI editor."""
    out = {
        "@type": "prov:Activity",
        "prov:label": label,
        "sc:description": description,
    }
    activity_uri = ACTIVITY_TYPE_URIS.get(activity_type)
    out["prov:type"] = {"@id": activity_uri} if activity_uri else activity_type
    if agents:
        out["prov:wasAttributedTo"] = agents
    return out


# Croissant 1.1 plus the NeurIPS D&B Responsible AI minimal fields.
CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "prov": "http://www.w3.org/ns/prov#",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "samplingRate": "cr:samplingRate",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


def aggregate_manifest():
    """Croissant manifest for the full AGC-Bench release."""
    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CROISSANT_CONFORMS_TO,
        "name": "AGC-Bench",
        "alternateName": "Artificial General Creativity Benchmark",
        "description": (
            "A meta-benchmark for evaluating creative ability in large "
            "language models, built from a PRISMA-compliant systematic "
            "review of 3,101 candidate papers (2018-2025). Comprises a "
            "curated catalog of 497 deduplicated benchmark records, of "
            "which 432 are creativity-relevant candidates, a runnable "
            "HELM-style evaluation harness for 78 onboarded benchmarks "
            "(67 text-only across six theory-driven domains plus 11 "
            "multimodal), aggregate per-(model, dataset) scores for 83 "
            "release-set language models that satisfy strict coverage "
            "criteria, "
            "and validation analyses including a "
            "c-factor extraction, separability test against fluid reasoning, "
            "MuCE convergent-validity test, and a paired human-LLM "
            "comparison via the Creativity Assessment Platform. Version "
            "1.0.1 adds catalog and onboarding records; scores, validation "
            "artifacts, and benchmark implementations are unchanged."
        ),
        "url": PROJECT_URL,
        "version": AGGREGATE_METADATA_VERSION,
        "datePublished": "2026-04-30",
        "dateModified": "2026-05-12",
        "releaseNotes": (
            "v1.0.1 adds release_data/benchmark_catalog_497.csv, catalog and "
            "onboarding records under curation/, the hosted generation-corpus "
            "layout, release-set dataset metadata, and a compact Dataset Viewer "
            "score table. Scores, validation artifacts, and benchmark "
            "implementations are unchanged."
        ),
        "license": LICENSE_URL,
        "citeAs": (
            "@inproceedings{agcbench2026,\n"
            "  title={AGC-Bench: Benchmarking Artificial General "
            "Creativity},\n"
            "  author={Anonymous Authors},\n"
            "  booktitle={NeurIPS 2026 Evaluations and Datasets Track},\n"
            "  year={2026}\n"
            "}"
        ),
        "creator": {
            "@type": "Organization",
            "name": "Anonymous AGC-Bench authors"
        },
        "keywords": [
            "creativity", "evaluation", "language models",
            "benchmark", "psychometrics", "factor analysis"
        ],

        # Responsible AI minimal set (NeurIPS D&B 2026 requirement)
        "rai:dataCollection": (
            "Catalog assembled via Semantic Scholar harvest (n=3,101 "
            "candidate papers), GPT-4.1 inclusion-rubric pre-screening "
            "(683 retained), Gemini 2.5 Flash data-availability "
            "verification (431 verified), four-reviewer dual human "
            "review under rotating assignment with lead-author conflict "
            "resolution, Gemini 2.5 Pro benchmark extraction (546 "
            "benchmarks across 1,160 tasks), and 10-pass Gemini "
            "consensus deduplication (497 unique benchmark records). A "
            "creativity-relevance pass retained 432 candidates and filtered "
            "65 adjacent/general co-extractions. The release includes "
            "the catalog-stage files for extracted benchmark mentions, "
            "deduplication output, and the relevance-filtered catalog. "
            "Scenario onboarding used a staged workflow with verbatim prompt "
            "extraction, source-paper scoring metric preservation, and human "
            "review at every accept step. Release-set evaluation against proprietary "
            "APIs and OpenRouter-served open-weight models with reasoning "
            "disabled by default for release-set comparability."
        ),
        "rai:dataPreprocessingProtocol": (
            "Per-(model, dataset) cells scored on each source paper's "
            "canonical metric. Within each dataset, raw metric values "
            "z-scored across the release set. Cells with multiple canonical "
            "metrics mean-aggregated. AGC-Human paired-human subset "
            "scored on the CrPO panel (diversity, DSI, surprise) with "
            "length-residualized variants for verbosity-confound control. "
            "Frozen seeded selection of n=50 items per (model, dataset) "
            "cell."
        ),
        "rai:dataLimitations": (
            "Release-1 is English-dominant. Multimodal coverage is "
            "limited to image-input tasks (video, audio, image-output "
            "out of scope). The primary composite is release-set-relative "
            "with no theoretical ceiling on the open-ended subset. "
            "Several canonical metrics use LLM-as-judge scoring with "
            "provider-specific judge models (data-quality audit reports "
            "kappa = 0.67 inter-judge agreement). 13 of 78 primary "
            "datasets do not write per-instance score outputs in the "
            "current run logs (cell-level only). The c-factor is "
            "estimated via exploratory factor analysis (EFA) on six "
            "domain composites with n=83 release-set models, sufficient "
            "for unidimensionality confirmation but not for confirmatory "
            "factor analysis with full fit indices."
        ),
        "rai:dataBiases": (
            "Source benchmarks reflect the distribution of creativity "
            "tasks the AI/LLM literature has produced, which over-samples "
            "Western, English-language, text-only constructs. The "
            "Story / Narrative domain is the largest by dataset count "
            "(21 of the 67 primary text-only datasets under the "
            "6-domain partition); robustness checks confirm the c-factor "
            "structure is preserved when this domain is downsampled, "
            "dropped, or merged with Figurative Language under the "
            "predecessor 5-domain partition. Judge-model selections may carry biases "
            "of their training data; the audit reports judge-canonical "
            "convergence at pooled r=0.37 with per-dataset median r=0.49."
        ),
        "rai:personalSensitiveInformation": (
            "AGC-Human paired-human subset contains anonymized human "
            "responses on five creativity tasks collected under prior "
            "approved IRB protocol (institution and study ID withheld "
            "for double-blind submission). No personally identifying "
            "information is released. The model-output corpus does not "
            "relate to identifiable people."
        ),
        "rai:dataUseCases": (
            "Intended uses: (1) systematic evaluation of creative ability "
            "in language models across content domains; (2) c-factor "
            "estimation for new models; (3) studies of mechanism "
            "tracing how training data, architecture, and post-training "
            "shape creative ability; (4) studies of enhancement "
            "examining prompting, fine-tuning, decoding strategy, and "
            "retrieval. Validity evidence from c-factor extraction "
            "(alpha=0.96, 81.5% variance, parallel-analysis confirmed), "
            "MuCE judgment convergent validity (Pearson r=0.66, Spearman "
            "rho=0.73 cross-model), and intervention sensitivity "
            "(be-creative dz=+1.40 length-residualized). Not validated "
            "for: clinical, educational, or employment decisions; "
            "ranking individual humans for creativity in deployment "
            "settings; absolute creativity scores on the open-ended "
            "subset (release-set-relative only)."
        ),
        "rai:dataSocialImpact": (
            "Standardizing creativity evaluation supports more rigorous "
            "comparison across models, methods, and time. Risks include "
            "narrowing the operational definition of creativity to what "
            "the assembled benchmarks measure; the appendix per-dataset "
            "table makes the operational definition fully transparent. "
            "Mitigations: documenting source-paper scoring conventions, "
            "preserving per-benchmark licenses, and distinguishing "
            "general release-set claims (release-set-relative) from absolute "
            "claims (closed-ended subset only)."
        ),
        "rai:dataReleaseDate": "2026-04-30",
        "rai:dataModality": ["text", "image"],
        "rai:dataAnnotationProtocol": (
            "Each (model, dataset) cell is annotated by the source paper's "
            "canonical scoring metric, applied verbatim from the source "
            "paper's released materials. Three canonical-metric families "
            "are represented: formula-based (exact-match, F1, ROUGE, BLEU, "
            "vote counts, parse rates), LLM-as-judge (Likert / wide-range "
            "rubric ratings under the source paper's prompt), and "
            "model-based semantic-distance (sentence-BERT, DSI, "
            "entailment). For the 24 LLM-judge cells, three frontier "
            "judges (Gemini-3-Flash, Grok-4.1-Fast, GPT-4.1-Mini) rate "
            "each item under a planned-missing 2-of-3 design; per-judge "
            "severity is removed via a Bayesian Graded Response Model "
            "(Judge Response Theory) before z-scoring. The AGC-Human "
            "paired-human subset uses the canonical CrPO scoring panel "
            "(diversity, DSI, surprise) computed at the response level."
        ),
        "rai:annotationsPerItem": (
            "Each model response in the primary release set receives one "
            "canonical-metric annotation. Items in the 24 LLM-judge cells "
            "additionally receive 2 of 3 LLM-judge ratings under the "
            "planned-missing JRT design (~91k rater-item observations "
            "total). AGC-Human paired-human responses are scored on three "
            "CrPO panel components per response. Judge ratings are "
            "produced by LLM judges (Gemini-3-Flash, Grok-4.1-Fast, "
            "GPT-4.1-Mini) rather than by human raters; the AGC-Human "
            "subset is the only component with human-produced responses, "
            "and its scores are computed by formula metrics, not "
            "synthesized."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "AGC-Bench is hosted on HuggingFace at "
            "https://huggingface.co/datasets/agcbench-2026/AGC-Bench with "
            "no time-limited retention restriction. The maintainers commit "
            "to keeping released artifacts publicly accessible indefinitely. "
            "Planned future updates extend the release set with new frontier "
            "models, onboard additional benchmarks from the 432-candidate "
            "creativity-relevant catalog, and add multimodal primary-result "
            "analyses; future updates will appear as separate version tags, "
            "with the present release preserved as a stable reference."
        ),
        "prov:wasDerivedFrom": _source_dataset_entries(_aggregate_source_datasets()),
        "prov:wasGeneratedBy": [
            _prov_activity(
                "Systematic source discovery and screening",
                "Data Collection",
                (
                    "PRISMA-compliant systematic literature harvest of 3,101 "
                    "candidate papers across 99 venues (January 2018-December "
                    "2025) drawn from Semantic Scholar. Inclusion-rubric "
                    "pre-screening with GPT-4.1 retained 683 papers; "
                    "Gemini-2.5-Flash with Google Search grounding verified "
                    "data accessibility for 431. Four trained reviewers "
                    "performed dual human review under a rotating-pair "
                    "protocol with lead-author conflict resolution; 283 "
                    "papers passed. Reviewers are English-proficient academic "
                    "researchers; institutional affiliation and geography are "
                    "withheld for double-blind review."
                ),
                agents=[
                    _agent("AGC-Bench review workflow"),
                    _agent("Semantic Scholar API", software=True),
                    _agent("GPT-4.1", software=True),
                    _agent("Gemini-2.5-Flash with Google Search grounding", software=True),
                ],
            ),
            _prov_activity(
                "Benchmark extraction, deduplication, and modality triage",
                "Data preprocessing",
                (
                    "Gemini-2.5-Pro extracted 546 candidate benchmark "
                    "mentions from 283 papers, producing "
                    "structured records (name, modality, task description, "
                    "evaluation metric, dataset URL). String-normalized "
                    "deduplication followed by 10-pass Gemini-2.5-Pro "
                    "consensus voting (>=80% inter-pass agreement) yielded "
                    "497 unique benchmark records. Relevance pass removed "
                    "65 adjacent/general NLP, computer-vision, reasoning, "
                    "intelligence, and auxiliary co-extractions, leaving "
                    "432 creativity-relevant candidates. Tier classification "
                    "by Gemini-3-Flash partitioned the catalog by "
                    "input/output modality and infrastructure requirement; "
                    "78 benchmarks were onboarded for the present release."
                ),
                agents=[
                    _agent("Gemini-2.5-Pro", software=True),
                    _agent("Gemini-3-Flash", software=True),
                    _agent("AGC-Bench review workflow"),
                ],
            ),
            _prov_activity(
                "Scenario onboarding and quality review",
                "Quality review",
                (
                    "Each onboarded benchmark was implemented as a HELM-style "
                    "scenario using the source paper's prompt template, "
                    "generation configuration, and canonical metric wherever "
                    "available. Scripts and audit notes were reviewed for "
                    "source fidelity, runnable setup, license metadata, and "
                    "data-quality flags before inclusion in the release set."
                ),
                agents=[
                    _agent("AGC-Bench implementation workflow"),
                    _agent("AGC-Bench review workflow"),
                ],
            ),
            _prov_activity(
                "Model response generation",
                "Data Collection",
                (
                    "Model responses on each benchmark are synthetic by "
                    "construction: generations from the 83 release-set "
                    "language models. Generations were produced under each "
                    "source paper's originating sampling configuration where "
                    "specified, with standardized defaults otherwise (n=50 "
                    "items per cell from a frozen seeded selection; smaller "
                    "datasets exhaustive)."
                ),
                agents=[
                    _agent("83 release-set language models", software=True),
                    _agent("AGC-Bench evaluation harness", software=True),
                ],
            ),
            _prov_activity(
                "Scoring, judge calibration, and release aggregation",
                "Annotation",
                (
                    "Each (model, dataset) cell is annotated by the source "
                    "paper's canonical metric. The 24 LLM-judge cells are "
                    "additionally rated by three frontier vendor judges "
                    "(google/gemini-3-flash-preview, x-ai/grok-4.1-fast, "
                    "openai/gpt-4.1-mini) under a planned-missing 2-of-3 "
                    "design at 50 items per cell, producing 182,924 ratings "
                    "on 90,987 units. Per-judge severity is removed via a "
                    "Bayesian Graded Response Model (Judge Response Theory). "
                    "Independent on-task data-quality audit by "
                    "x-ai/grok-4.1-fast on three random items per cell "
                    "yielded 95.1% on-task rate. The AGC-Human paired-human "
                    "subset is scored on the canonical CrPO panel "
                    "(diversity, DSI, length-residualized z) computed at "
                    "the response level."
                ),
                agents=[
                    _agent("AGC-Bench scoring scripts", software=True),
                    _agent("google/gemini-3-flash-preview", software=True),
                    _agent("x-ai/grok-4.1-fast", software=True),
                    _agent("openai/gpt-4.1-mini", software=True),
                ],
            ),
        ],
        "rai:hasSyntheticData": True,

        # Distribution: aggregate score artifacts, validation analyses,
        # release curation metadata, and the Hugging Face generation corpus
        # file layout.
        "distribution": (
            [_file_object(spec) for spec in DISTRIBUTION_FILES]
            + [_file_set(spec) for spec in DISTRIBUTION_FILESETS]
        ),

        # RecordSet: structured schema for the catalog
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "catalog-records",
                "name": "catalog",
                "description": "One row per unique creativity benchmark in the catalog.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/benchmark_id",
                        "name": "benchmark_id",
                        "description": "Stable benchmark identifier.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "benchmark_id"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/source_paper_url",
                        "name": "source_paper_url",
                        "description": "DOI or arXiv URL of the source paper.",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "source_paper_url"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/domain",
                        "name": "domain",
                        "description": (
                            "Primary creativity domain in {Brainstorming, "
                            "Problem Solving, STEM, Literary and "
                            "Narrative, Humor, Visual and Design}."
                        ),
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "domain"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/modality",
                        "name": "modality",
                        "description": "Tier classification (Tier-1a text-only, Tier-1b image-input, etc.).",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "modality"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/scoring_metric",
                        "name": "scoring_metric",
                        "description": "Canonical scoring metric specified by the source paper.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "scoring_metric"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/license",
                        "name": "license",
                        "description": "Source-paper license for the underlying benchmark data.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "license"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog-records/onboarded",
                        "name": "onboarded",
                        "description": "Whether the benchmark is in the released onboarded set.",
                        "dataType": "sc:Boolean",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog"},
                            "extract": {"column": "onboarded"}
                        }
                    },
                ]
            },
            {
                "@type": "cr:RecordSet",
                "@id": "catalog497-records",
                "name": "benchmark_catalog_497",
                "description": (
                    "One row per deduplicated PRISMA benchmark record. "
                    "Rows with creativity_relevant=true are the 432 "
                    "creativity-relevant onboarding candidates; the "
                    "remaining 65 adjacent/general records are retained for "
                    "auditability."
                ),
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/benchmark_name",
                        "name": "benchmark_name",
                        "description": "Canonical deduplicated benchmark name.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "benchmark_name"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/aliases",
                        "name": "aliases",
                        "description": "Semicolon-delimited benchmark aliases merged into this record.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "aliases"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/creativity_relevant",
                        "name": "creativity_relevant",
                        "description": "Whether the record belongs to the 432 creativity-relevant candidate subset.",
                        "dataType": "sc:Boolean",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "creativity_relevant"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/source_paper_title",
                        "name": "source_paper_title",
                        "description": "Source paper title for the first extracted occurrence of the benchmark.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "source_paper_title"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/source_paper_year",
                        "name": "source_paper_year",
                        "description": "Source paper publication year.",
                        "dataType": "sc:Integer",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "source_paper_year"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/source_paper_doi",
                        "name": "source_paper_doi",
                        "description": "Source paper DOI where available.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "source_paper_doi"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/source_paper_semantic_scholar_id",
                        "name": "source_paper_semantic_scholar_id",
                        "description": "Semantic Scholar paper identifier used in the curation pipeline.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "source_paper_semantic_scholar_id"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/modality",
                        "name": "modality",
                        "description": "Input/output modality assigned for creativity-relevant records.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "modality"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/runnability_tier",
                        "name": "runnability_tier",
                        "description": "Implementation/runnability tier assigned for creativity-relevant records.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "runnability_tier"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/scoring_method",
                        "name": "scoring_method",
                        "description": "Scoring-method family extracted from the source paper.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "scoring_method"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/scoring_protocol",
                        "name": "scoring_protocol",
                        "description": "Extracted source-paper scoring protocol details.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "scoring_protocol"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "catalog497-records/task_types",
                        "name": "task_types",
                        "description": "Extracted task-type families associated with the benchmark.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "agc-bench-catalog-497"},
                            "extract": {"column": "task_types"}
                        }
                    },
                ]
            }
        ]
    }


def per_benchmark_manifest(benchmark_id: str, info: dict) -> dict:
    """Croissant manifest for a single onboarded benchmark."""
    domain = DOMAIN_MAPPING.get(benchmark_id, "Unspecified")
    paper_url = info.get('source_paper', '')
    repo_url = info.get('source_repo', '')
    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CROISSANT_CONFORMS_TO,
        "name": f"AGC-Bench/{benchmark_id}",
        "alternateName": info.get('display_name', benchmark_id),
        "description": (
            f"Onboarded scenario for the source benchmark "
            f"'{info.get('display_name', benchmark_id)}' in the AGC-Bench "
            f"released model set. Domain: {domain}. Source paper: "
            f"{paper_url}. Source repository: {repo_url}. The AGC-Bench "
            f"release preserves the source paper's prompt template, "
            f"generation configuration, and canonical scoring metric. "
            f"This Croissant manifest documents the AGC-Bench scenario "
            f"file and the per-(model, item) outputs produced by the "
            f"release evaluation. The original benchmark license applies "
            f"to the underlying source data; the AGC-Bench scenario "
            f"code is released under Apache-2.0."
        ),
        "url": paper_url or repo_url or PROJECT_URL,
        "version": RELEASE_VERSION,
        "datePublished": "2026-04-30",
        "license": "See source paper for benchmark license; scenario code Apache-2.0.",
        "citeAs": (
            f"See source paper at {paper_url} for the original benchmark "
            f"citation."
        ),
        "creator": {
            "@type": "Organization",
            "name": "AGC-Bench (anonymous; camera-ready will identify the lab)"
        },
        "keywords": [
            "creativity", "evaluation", "language models", domain.lower()
        ],
        "rai:dataLimitations": (
            "Per-instance scoring fidelity depends on the source "
            "benchmark's evaluation protocol. The release set is "
            "frozen at the AGC-Bench submission cutoff; results may "
            "differ from later evaluations."
        ),
        "rai:dataBiases": (
            "Inherits any biases of the source benchmark's task design "
            "and reference annotations; consult the source paper for "
            "details."
        ),
        "rai:personalSensitiveInformation": (
            "The source benchmark's stance on personal/sensitive "
            "information is documented in the source paper. The "
            "AGC-Bench release does not introduce additional personal "
            "data."
        ),
        "rai:dataUseCases": (
            "Use as part of the AGC-Bench evaluation suite for "
            "creativity benchmarking, c-factor estimation, and "
            "intervention experiments. Not validated for: clinical, "
            "educational, or employment decisions; ranking individual "
            "humans for creativity in deployment settings."
        ),
        "rai:dataSocialImpact": (
            "This per-benchmark manifest inherits the social-impact profile "
            "of AGC-Bench as an evaluation-only research artifact. Positive "
            "uses include transparent comparison of model creativity on a "
            "documented source task. Risks include over-interpreting one "
            "benchmark as a complete measure of creativity or applying "
            "release-set-relative scores outside their validation context."
        ),
        "prov:wasDerivedFrom": _source_dataset_entries([u for u in (paper_url, repo_url) if u]),
        "prov:wasGeneratedBy": [
            _prov_activity(
                "Source benchmark selection",
                "Data Collection",
                (
                    f"Source benchmark drawn from the AGC-Bench PRISMA "
                    f"catalog (3,101 candidate papers screened, 497 unique "
                    f"benchmarks identified). This benchmark's primary "
                    f"reference is {paper_url or 'see source repo'}."
                ),
                agents=[_agent("Anonymous AGC-Bench curation team")],
            ),
            _prov_activity(
                "Per-item model scoring",
                "Annotation",
                (
                    "Per-(model, item) outputs are scored under the "
                    "source paper's canonical metric, applied verbatim "
                    "from the source paper's released materials. For "
                    "LLM-judge metrics, three frontier vendor judges "
                    "(google/gemini-3-flash-preview, x-ai/grok-4.1-fast, "
                    "openai/gpt-4.1-mini) rate each item under the "
                    "AGC-Bench planned-missing 2-of-3 design with "
                    "Bayesian Graded Response Model calibration "
                    "(Judge Response Theory)."
                ),
                agents=[
                    _agent("AGC-Bench scoring scripts", software=True),
                    _agent("LLM judges for JRT-scored cells", software=True),
                ],
            ),
            _prov_activity(
                "Per-item model response generation",
                "Data Collection",
                (
                    "Per-(model, item) responses on this benchmark are "
                    "synthetic by construction: language-model generations "
                    "from the AGC-Bench release-set evaluation."
                ),
                agents=[
                    _agent("AGC-Bench evaluation harness", software=True),
                    _agent("Release-set language models", software=True),
                ],
            ),
        ],
        "rai:hasSyntheticData": True,
        "distribution": _per_benchmark_distribution(benchmark_id),
    }


def _per_benchmark_distribution(benchmark_id):
    """Per-benchmark FileObjects: the scenario file (always shipped) and the
    canonical-score slice within release_data/long_model_x_dataset.csv. Per-(model,
    item) generation parquets are NOT redistributed, so they are not
    referenced here."""
    scenario_local = REPO / f'scenarios/{benchmark_id}_scenario.py'
    scenario_sha = _sha256(scenario_local) if scenario_local.exists() else 'missing-local-file'
    return [
        {
            '@type': 'cr:FileObject',
            '@id': f'{benchmark_id}-scenario',
            'name': f'{benchmark_id}_scenario.py',
            'description': f'AGC-Bench scenario file for {benchmark_id}.',
            'contentUrl': f'{DATASET_URL}/resolve/main/scenarios/{benchmark_id}_scenario.py',
            'encodingFormat': 'text/x-python',
            'sha256': scenario_sha,
        },
    ]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    PER_BENCH_DIR.mkdir(exist_ok=True)

    registry_master = yaml.safe_load(
        (REPO / 'data/registry/registry_master.yaml').read_text())['datasets']

    # Aggregate manifest
    agg = aggregate_manifest()
    agg_path = OUT_DIR / 'agc_bench.json'
    agg_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f'Wrote aggregate Croissant: {agg_path}')

    # Per-benchmark manifests for the released set. Filter to the benchmark IDs
    # that ship as scenarios in this bundle so the manifest set matches the
    # released artifacts (the predecessor DOMAIN_MAPPING also covers excluded
    # legacy benches that are not part of the release).
    released_benchmark_ids = sorted(
        p.stem.replace('_scenario', '')
        for p in (REPO / 'scenarios').glob('*_scenario.py')
    )
    n_written = 0
    n_skipped = 0
    n_no_domain = 0
    for benchmark_id in released_benchmark_ids:
        info = registry_master.get(benchmark_id)
        if info is None:
            n_skipped += 1
            continue
        if benchmark_id not in DOMAIN_MAPPING:
            n_no_domain += 1
            continue
        manifest = per_benchmark_manifest(benchmark_id, info)
        path = PER_BENCH_DIR / f'{benchmark_id}.json'
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        n_written += 1
    print(f'Wrote {n_written} per-benchmark Croissant manifests to {PER_BENCH_DIR}')
    if n_skipped:
        print(f'  ({n_skipped} release benchmark IDs had no registry entry; skipped)')
    if n_no_domain:
        print(f'  ({n_no_domain} release benchmark IDs had no domain assignment; skipped)')


if __name__ == '__main__':
    main()
