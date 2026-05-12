"""License-lookup pipeline for AGC-Bench v1.0 release.

For each of the 90 validated datasets, this script:
  1. Reads source_repo URL from registry_master.yaml
  2. Tries to fetch a LICENSE file from the repo (GitHub, HuggingFace)
  3. Heuristically classifies the license (MIT, Apache-2.0, CC-BY-*, BSD, etc.)
  4. Writes the result to audit/dataset_licenses.csv

For GitHub repos: uses the GitHub raw URL pattern to fetch LICENSE / LICENSE.md /
LICENSE.txt directly. For HuggingFace dataset repos, fetches the README.md and
extracts the YAML frontmatter `license:` field.

Output: audit/dataset_licenses.csv with columns:
  dataset, source_repo, license_url, license_label, license_text_excerpt, lookup_status

Run separately to update registry_master.yaml from the resolved licenses.
"""
from __future__ import annotations
import yaml, re, time, os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'audit'
OUT.mkdir(parents=True, exist_ok=True)


def github_raw_candidates(repo_url: str) -> list[str]:
    """Generate raw.githubusercontent.com URLs for typical LICENSE files."""
    m = re.match(r'https?://github\.com/([^/]+)/([^/?#]+)', repo_url)
    if not m:
        return []
    org, repo = m.group(1), m.group(2).rstrip('/').replace('.git', '')
    bases = []
    for branch in ('main', 'master'):
        for fn in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'LICENCE', 'COPYING'):
            bases.append(f'https://raw.githubusercontent.com/{org}/{repo}/{branch}/{fn}')
    return bases


def hf_dataset_readme(repo_url: str) -> str | None:
    """Return raw URL for a HuggingFace dataset README.md."""
    m = re.match(r'https?://huggingface\.co/datasets/([^/?#]+)/([^/?#]+)', repo_url)
    if not m:
        return None
    user, name = m.group(1), m.group(2)
    return f'https://huggingface.co/datasets/{user}/{name}/raw/main/README.md'


def hf_model_readme(repo_url: str) -> str | None:
    m = re.match(r'https?://huggingface\.co/(?!datasets/)([^/?#]+)/([^/?#]+)', repo_url)
    if not m:
        return None
    user, name = m.group(1), m.group(2)
    return f'https://huggingface.co/{user}/{name}/raw/main/README.md'


def fetch(url: str, timeout: int = 8) -> str | None:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'agc-bench-license-audit/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode(errors='replace')
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
        pass
    return None


LICENSE_PATTERNS = [
    ('MIT', re.compile(r'\bMIT License\b|Permission is hereby granted, free of charge', re.IGNORECASE)),
    ('Apache-2.0', re.compile(r'\bApache License,?\s*Version 2\.0\b|apache\.org/licenses/LICENSE-2\.0', re.IGNORECASE)),
    ('BSD-3-Clause', re.compile(r'\bBSD\b.*3-?Clause|Redistributions of source code must retain', re.IGNORECASE)),
    ('BSD-2-Clause', re.compile(r'\bBSD\b.*2-?Clause', re.IGNORECASE)),
    ('GPL-3.0', re.compile(r'\bGNU GENERAL PUBLIC LICENSE.*Version 3', re.IGNORECASE | re.DOTALL)),
    ('GPL-2.0', re.compile(r'\bGNU GENERAL PUBLIC LICENSE.*Version 2', re.IGNORECASE | re.DOTALL)),
    ('LGPL', re.compile(r'GNU LESSER GENERAL PUBLIC LICENSE', re.IGNORECASE)),
    ('AGPL-3.0', re.compile(r'GNU AFFERO GENERAL PUBLIC LICENSE', re.IGNORECASE)),
    ('CC-BY-4.0', re.compile(r'Creative Commons Attribution\s*4\.0|CC[ -]BY[ -]4\.0|CC-BY 4', re.IGNORECASE)),
    ('CC-BY-SA-4.0', re.compile(r'Creative Commons Attribution-ShareAlike\s*4\.0|CC[ -]BY[ -]SA[ -]4\.0', re.IGNORECASE)),
    ('CC-BY-NC-4.0', re.compile(r'Creative Commons Attribution-NonCommercial\s*4\.0|CC[ -]BY[ -]NC[ -]4\.0', re.IGNORECASE)),
    ('CC0-1.0', re.compile(r'CC0\s*1\.0|Creative Commons Zero', re.IGNORECASE)),
    ('Unlicense', re.compile(r'\bThe Unlicense\b|This is free and unencumbered software released into the public domain', re.IGNORECASE)),
    ('ODC-By-1.0', re.compile(r'Open Data Commons Attribution', re.IGNORECASE)),
]


def classify_license(text: str) -> str:
    if not text:
        return 'UNKNOWN'
    for label, pat in LICENSE_PATTERNS:
        if pat.search(text):
            return label
    return 'OTHER'


def lookup_one(benchmark_id: str, repo_url: str) -> dict:
    if not repo_url:
        return {'dataset': benchmark_id, 'source_repo': '', 'license_url': '',
                'license_label': 'NO_REPO_URL', 'license_text_excerpt': '', 'lookup_status': 'no-source-repo'}

    # Try GitHub raw LICENSE files
    for url in github_raw_candidates(repo_url):
        text = fetch(url)
        if text:
            label = classify_license(text)
            return {'dataset': benchmark_id, 'source_repo': repo_url, 'license_url': url,
                    'license_label': label, 'license_text_excerpt': text[:200].replace('\n', ' '),
                    'lookup_status': 'github-raw-found'}

    # Try GitHub API for license metadata
    m = re.match(r'https?://github\.com/([^/]+)/([^/?#]+)', repo_url)
    if m:
        org, repo = m.group(1), m.group(2).rstrip('/').replace('.git', '')
        api_url = f'https://api.github.com/repos/{org}/{repo}/license'
        text = fetch(api_url)
        if text:
            try:
                d = json.loads(text)
                spdx = d.get('license', {}).get('spdx_id', 'OTHER')
                return {'dataset': benchmark_id, 'source_repo': repo_url, 'license_url': api_url,
                        'license_label': spdx, 'license_text_excerpt': d.get('license', {}).get('name', ''),
                        'lookup_status': 'github-api'}
            except Exception:
                pass

    # Try HuggingFace dataset README
    readme_url = hf_dataset_readme(repo_url)
    if readme_url:
        text = fetch(readme_url)
        if text:
            # Look for YAML frontmatter `license:` field
            ymatch = re.search(r'^---\s*\n(.*?)\n---', text, re.DOTALL | re.MULTILINE)
            if ymatch:
                lic_match = re.search(r'^license:\s*(\S+)', ymatch.group(1), re.MULTILINE)
                if lic_match:
                    return {'dataset': benchmark_id, 'source_repo': repo_url, 'license_url': readme_url,
                            'license_label': lic_match.group(1).strip(),
                            'license_text_excerpt': ymatch.group(1)[:200].replace('\n', ' '),
                            'lookup_status': 'hf-dataset-readme'}
            return {'dataset': benchmark_id, 'source_repo': repo_url, 'license_url': readme_url,
                    'license_label': 'UNKNOWN_HF', 'license_text_excerpt': '',
                    'lookup_status': 'hf-readme-no-license-field'}

    # Try HuggingFace model README
    readme_url = hf_model_readme(repo_url)
    if readme_url:
        text = fetch(readme_url)
        if text:
            ymatch = re.search(r'^---\s*\n(.*?)\n---', text, re.DOTALL | re.MULTILINE)
            if ymatch:
                lic_match = re.search(r'^license:\s*(\S+)', ymatch.group(1), re.MULTILINE)
                if lic_match:
                    return {'dataset': benchmark_id, 'source_repo': repo_url, 'license_url': readme_url,
                            'license_label': lic_match.group(1).strip(),
                            'license_text_excerpt': ymatch.group(1)[:200].replace('\n', ' '),
                            'lookup_status': 'hf-model-readme'}

    return {'dataset': benchmark_id, 'source_repo': repo_url, 'license_url': '',
            'license_label': 'NOT_FOUND', 'license_text_excerpt': '',
            'lookup_status': 'all-strategies-failed'}


def main():
    master = yaml.safe_load(open(REPO / 'data/registry/registry_master.yaml'))['datasets']
    long_path = REPO / 'release_data/long_model_x_dataset.csv'
    text_only = sorted(pd.read_csv(long_path)['dataset'].unique().tolist())
    mm = ['rebus_puzzle','banner_request_400','mars','hummus','irfl','ii_bench',
          'infochartqa','puzzleworld','creation_mmbench','artinsight','esp_dataset','yesbut']
    release_benchmarks = text_only + mm

    print(f'Looking up licenses for {len(release_benchmarks)} datasets...')

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {
            ex.submit(lookup_one, benchmark_id, master.get(benchmark_id, {}).get('source_repo', '')): benchmark_id
            for benchmark_id in release_benchmarks
        }
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(release_benchmarks)} done, latest: {r["dataset"]} -> {r["license_label"]}')

    df = pd.DataFrame(results).sort_values('dataset')
    df.to_csv(OUT / 'dataset_licenses.csv', index=False)
    print(f'\nWrote {OUT/"dataset_licenses.csv"}')

    print(f'\n=== License distribution ===')
    print(df['license_label'].value_counts().to_string())

    print(f'\n=== Lookup status ===')
    print(df['lookup_status'].value_counts().to_string())

    not_found = df[df['license_label'].isin(['NOT_FOUND', 'NO_REPO_URL', 'UNKNOWN_HF'])]
    if len(not_found):
        print(f'\nDatasets needing manual license lookup ({len(not_found)}):')
        for _, r in not_found.iterrows():
            print(f'  {r["dataset"]:<35} {r["source_repo"]}')


if __name__ == '__main__':
    main()
