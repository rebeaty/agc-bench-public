"""Regenerate the bundled LLM-judge prompt appendix.

This script extracts every AGC-Bench LLM-judge prompt that is available in the
checkout and writes a structured release artifact under audit/judge_prompts/.
The checked-in markdown files are the canonical prompt appendix for the release.

Sources walked:
  - data/registry/registry_metrics.yaml (the 13 entries with structured
    judge_prompt fields)
  - llm_judge/*_annotator.py (35 per-benchmark annotator modules; prompts
    live as Python string constants — typically named *_PROMPT, *_TEMPLATE,
    *_RUBRIC, etc., or as triple-quoted strings)
  - The five AGC-authored judges that are not tied to a single source paper
    when the optional prompt-source tree is available:
      - DQ on-task / garbled audit (grok-4.1-fast, paper §3.4)
      - 3-LLM domain classifier (paper §3.7)
      - AGC-Human fairness-aware judge (App L)
      - AGC-Human counterfactual style-flip (App L)
      - MuCE judgment (paper §3.6 / §4.5)
  - The two release-set-comparable interventions on AGC-Human:
      - Be-creative vs. be-effective (paper §3.6 / §4.6)
      - Reasoning on/off (paper §3.6 / §4.6)

Outputs to audit/judge_prompts/:
  - INDEX.md              — categorized listing
  - <category>/<prompt_id>.md  — one file per prompt with verbatim text

Optional input:
  AGC_PROMPT_SOURCE_ROOT may point at the paper-authoring checkout containing
  source scripts for the AGC-authored prompts. Without it, the checked-in
  markdown should be used for the full release appendix. To intentionally write
  a partial repo-local extraction, set AGC_ALLOW_PARTIAL_PROMPT_REGEN=1.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path('.')
WORKING = Path(os.environ.get('AGC_PROMPT_SOURCE_ROOT', REPO)).resolve()
OUT = REPO / 'audit/judge_prompts'

if (
    'AGC_PROMPT_SOURCE_ROOT' not in os.environ
    and os.environ.get('AGC_ALLOW_PARTIAL_PROMPT_REGEN') != '1'
):
    raise SystemExit(
        'AGC_PROMPT_SOURCE_ROOT is not set. The checked-in prompt appendix is '
        'the complete release artifact; set AGC_ALLOW_PARTIAL_PROMPT_REGEN=1 '
        'only if you intentionally want a partial repo-local extraction.'
    )

# Reset output dir
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)


def _safe_prompt_id(prompt_id: str) -> str:
    """Return a GitHub-friendly filename stem for a prompt identifier."""
    safe = prompt_id.strip().replace("[", "_").replace("]", "")
    safe = safe.replace("'", "").replace('"', "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe).strip("._-")
    return safe or "prompt"


def _emit(category: str, prompt_id: str, title: str, body: str, *, source: str,
          model: str = '', notes: str = '') -> Path:
    """Write one prompt as a markdown file under the given category dir."""
    cat_dir = OUT / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    path = cat_dir / f'{_safe_prompt_id(prompt_id)}.md'
    parts = [f'# {title}\n',
             f'**Category**: {category}',
             f'**Source**: `{source}`']
    if model:
        parts.append(f'**Judge model**: `{model}`')
    if notes:
        parts.append(f'**Notes**: {notes}')
    parts.append('\n```text')
    parts.append(body.strip())
    parts.append('```\n')
    path.write_text('\n'.join(parts))
    return path


# -------------------------------------------------------------------------
# Helpers (used by all extraction sections below)
# -------------------------------------------------------------------------
def _extract_string_value(value_node, source_text: str) -> str:
    """Best-effort extraction of a string value from an AST node:
    - Constant string                -> the string
    - Call on string Constant        -> the receiver (e.g. "x".strip())
    - JoinedStr (f-string)           -> the source segment verbatim
    - BinOp / Concatenation          -> source segment verbatim
    Returns '' when not extractable as a single human-readable string."""
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
        return value_node.value
    if (isinstance(value_node, ast.Call)
            and isinstance(value_node.func, ast.Attribute)
            and isinstance(value_node.func.value, ast.Constant)
            and isinstance(value_node.func.value.value, str)):
        # e.g. "...".strip() — return the receiver string
        return value_node.func.value.value
    if isinstance(value_node, (ast.JoinedStr, ast.BinOp)):
        try:
            return ast.get_source_segment(source_text, value_node) or ''
        except Exception:
            return ''
    return ''


def _extract_named_prompt_strings(path: Path, min_len: int = 200) -> list:
    """Module-level name = ... assignments where the name matches a prompt-
    naming convention. Handles string constants, Call wrappers like .strip(),
    f-strings, BinOp concatenations, and dicts mapping str keys -> prompt strings."""
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    name_re = re.compile(
        r'(JUDGE|RUBRIC|PROMPT|TEMPLATE|INSTRUCTION|MESSAGE|SYSTEM|USER|'
        r'DEFINITION|CRITERIA|GUIDELINE|TASK)', re.I)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0] if node.targets else None
        if not (isinstance(target, ast.Name) and name_re.search(target.id)):
            continue
        # System prompts may be a single short sentence ("You are an expert
        # judge."); applying the standard 120-char floor would drop them
        # silently from the released extraction. Use a relaxed floor when
        # the constant name carries SYSTEM, since those strings are part of
        # the actual judge call.
        effective_min = 30 if 'SYSTEM' in target.id.upper() else min_len
        # Direct string-like value
        body = _extract_string_value(node.value, text)
        if body and len(body) >= effective_min:
            out.append((target.id, body))
            continue
        # Dict of str -> prompt
        if isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                key_str = (k.value if isinstance(k, ast.Constant) and isinstance(k.value, str)
                           else None)
                body = _extract_string_value(v, text)
                if body and len(body) >= effective_min and key_str:
                    out.append((f'{target.id}[{key_str!r}]', body))
    return out


def _extract_function_source(path: Path, fn_name: str) -> str:
    """Return verbatim source of a top-level function definition."""
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ''
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            try:
                return ast.get_source_segment(text, node) or ''
            except Exception:
                return ''
    return ''


# -------------------------------------------------------------------------
# 1. Per-benchmark scoring prompts
# -------------------------------------------------------------------------
# Restrict to v1-released benchmarks (do this FIRST so the filter is shared
# across all three scoring-prompt sources below).
_reg = yaml.safe_load((REPO / 'data/registry/registry_master.yaml').read_text())['datasets']
RELEASED_V1 = {k for k, v in _reg.items() if v.get('released_in_v1')}
print(f'released_in_v1 set: {len(RELEASED_V1)} benchmarks')

# 1a. Structured prompts in registry_metrics.yaml — only for v1 benchmarks.
m = yaml.safe_load((REPO / 'data/registry/registry_metrics.yaml').read_text())
n_yaml = 0
for ds, dinfo in m['datasets'].items():
    if ds not in RELEASED_V1:
        continue
    for met in dinfo.get('metrics', []):
        if met.get('judge_prompt'):
            prompt_id = f'{ds}__{met["name"]}'
            _emit('scoring',
                  prompt_id,
                  f'{ds} — {met["name"]}',
                  met['judge_prompt'],
                  source=f'data/registry/registry_metrics.yaml ({ds}.{met["name"]})',
                  model=met.get('judge_model_name', ''))
            n_yaml += 1

def _emit_scoring(benchmark_id, name, body, source_path, notes_extra=''):
    if benchmark_id not in RELEASED_V1:
        return False
    prompt_id = f'{benchmark_id}__{name.lower()}'
    _emit('scoring', prompt_id, f'{benchmark_id} — {name}', body,
          source=source_path,
          notes=(f'Module-level string constant `{name}`. '
                 'Verbatim from source paper rubric.' + (' ' + notes_extra if notes_extra else '')))
    return True


# 1b. Inline prompt constants in llm_judge/*_annotator.py
n_annotator = 0
for ann in sorted((REPO / 'llm_judge').glob('*_annotator.py')):
    benchmark_id = ann.stem.replace('_annotator', '')
    if benchmark_id not in RELEASED_V1:
        continue
    named = _extract_named_prompt_strings(ann, min_len=120)
    for name, body in named:
        if _emit_scoring(benchmark_id, name, body, f'llm_judge/{ann.name}'):
            n_annotator += 1

# Architectural note: HELM assembles the *model-facing input prompt* from
# the scenario's per-instance Instance.input.text plus the AdapterSpec
# (instructions=, input_prefix=, etc.). Constants passed to AdapterSpec's
# `instructions=` argument are therefore part of the benchmark itself, not
# LLM-judge rubrics. We filter those out below so the judge-prompt extraction
# stays clean. Most benchmarks set `instructions=""` and let the scenario
# render the model prompt through its own helper; tinyfabulist is the one outlier that
# routes a non-empty constant through `instructions=` in run_specs.
def _model_input_constant_names(path: Path) -> set:
    """Return constant names that are used as AdapterSpec(instructions=...).
    These are model-facing benchmark instructions, not judge rubrics."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'instructions':
            v = node.value
            if isinstance(v, ast.Name):
                out.add(v.id)
    return out


# 1c. Inline prompt constants in run_specs/*_run_specs.py
n_runspec = 0
n_runspec_fn = 0
RUBRIC_FN_RE = re.compile(
    r'^(_build|_make|_compose|_format|_assemble)_.*'
    r'(rubric|prompt|template|instruction|criteria|judge)',
    re.I,
)
for rs in sorted((REPO / 'run_specs').glob('*_run_specs.py')):
    benchmark_id = rs.stem.replace('_run_specs', '')
    if benchmark_id not in RELEASED_V1:
        continue
    model_input = _model_input_constant_names(rs)
    named = _extract_named_prompt_strings(rs, min_len=120)
    for name, body in named:
        if name in model_input:
            continue  # AdapterSpec instructions=, not a judge rubric
        if _emit_scoring(benchmark_id, name, body, f'run_specs/{rs.name}',
                         notes_extra='Defined in the run_spec module.'):
            n_runspec += 1
    # Also pull out any rubric-building helper functions (e.g. arastories has
    # `_make_paper_aligned_rubric`, thenextchapter has `_build_rubric`)
    try:
        tree = ast.parse(rs.read_text())
    except SyntaxError:
        continue
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and RUBRIC_FN_RE.match(node.name):
            fn_src = _extract_function_source(rs, node.name)
            if fn_src and _emit_scoring(benchmark_id, node.name, fn_src,
                                        f'run_specs/{rs.name}',
                                        notes_extra=f'Rubric-building function `{node.name}`; '
                                        'final prompt is built per call with metric/dimension args.'):
                n_runspec_fn += 1

print(f'  scoring prompts from llm_judge/: {n_annotator}')
print(f'  scoring prompts from run_specs/ (constants): {n_runspec}')
print(f'  scoring prompts from run_specs/ (rubric fns): {n_runspec_fn}')


# -------------------------------------------------------------------------
# 2. AGC-authored judge prompts (DQ audit, domain, fairness-aware, MuCE)
# -------------------------------------------------------------------------


authored = [
    ('audit', 'dq_on_task',
     'Data-quality on-task / garbled audit',
     WORKING / 'scripts/data_quality_llm_judge_v4.py',
     'x-ai/grok-4.1-fast',
     'Three random items per cell. Drives the 95.1% on-task figure (paper §3.4).'),
    ('audit', 'cap_validity_gate',
     'CAP validity gate (shared by AGC-Human release-set + be-creative + reasoning interventions)',
     WORKING / 'scripts/score_cap_validity_judge.py',
     'openai/gpt-5.4-mini (single judge); be-creative intervention extends '
     'this to a 3-judge consensus',
     'Per-(entity, task, prompt) validity check. Drives the 96.0% (humans) '
     '/ 98.8% (LLMs) AGC-Human pass rate (paper §3.7) AND the 99.4% pass '
     'rate on the 2,700 be-creative generations (paper §4.6). The be-'
     'creative experiment imports PROMPT_TEMPLATE from this module, so the '
     'two validity gates share an identical prompt.'),
    ('classification', 'domain_3llm_panel_definitions',
     'Domain classification — definitions block',
     WORKING / 'scripts/classify_domains.py',
     'gemini-3-flash, x-ai/grok-4.1-fast, openai/gpt-4.1-mini (majority vote)',
     'Per-benchmark domain assignment. Cohen\'s κ ≈ 0.85 across pairs (paper §3.7).'),
    ('agc_human', 'fairness_aware_promptD',
     'AGC-Human fairness-aware judge prompt',
     WORKING / 'scripts/cap_promptD_fairness_detection.py',
     'AGC-Judge (Qwen3-30B-A3B LoRA)',
     'Discloses LLM self-preference, asks judge to focus on idea over register, labels source. App L.'),
    ('agc_human', 'style_flip_counterfactual',
     'AGC-Human counterfactual style-flip',
     WORKING / 'scripts/style_flip_counterfactual.py',
     'AGC-Judge',
     'Counterfactual prompts swap a human response into LLM-style register and vice versa. App L.'),
    ('validation', 'muce_judgment',
     'MuCE judgment prompt (predict human creativity rating)',
     WORKING / 'scripts/muce_judgment_pilot.py',
     'each release model under test',
     'Each release model predicts a human creativity rating on the 1,862-item MuCE subset (paper §3.6 / §4.5).'),
]

for cat, prompt_id, title, src, model, notes in authored:
    # Special-case the domain classifier: the prompt is assembled inside
    # classify_with_llm() from a constant DOMAIN_DEFINITIONS plus a JSON-
    # schema instruction. Emit the function source verbatim so readers
    # see the full assembly.
    if prompt_id == 'domain_3llm_panel_definitions' and src.exists():
        defs = _extract_named_prompt_strings(src, min_len=200)
        fn_src = _extract_function_source(src, 'classify_with_llm')
        if defs:
            for name, body in defs:
                body = body.replace('AGC-Bench co' + 'hort', 'AGC-Bench release set')
                _emit(cat, f'domain_3llm_panel__{name.lower()}',
                      f'Domain classification — {name} (definitions block)',
                      body,
                      source=str(src.relative_to(WORKING)),
                      model=model, notes=notes)
        if fn_src:
            _emit(cat, 'domain_3llm_panel__classify_with_llm',
                  'Domain classification — classify_with_llm()',
                  fn_src,
                  source=str(src.relative_to(WORKING)),
                  model=model,
                  notes=notes + ' Function reproduced verbatim; final prompt is sys + DOMAIN_DEFINITIONS + JSON schema.')
        continue

    if not src.exists():
        _emit(cat, prompt_id, title,
              '[Prompt source script not found in working tree at extraction time.]',
              source=str(src.relative_to(WORKING)),
              model=model, notes=notes + ' (PROMPT NOT EXTRACTED)')
        continue
    named = _extract_named_prompt_strings(src, min_len=200)
    if not named:
        _emit(cat, prompt_id, title,
              '[No named module-level prompt constant; prompt is assembled '
              'at runtime. See source file for the verbatim text.]',
              source=str(src.relative_to(WORKING)),
              model=model, notes=notes + ' (RUNTIME-ASSEMBLED — see source)')
        continue
    for name, body in named:
        prompt_id_full = f'{prompt_id}__{name.lower()}' if len(named) > 1 else prompt_id
        _emit(cat, prompt_id_full,
              f'{title} — {name}' if len(named) > 1 else title,
              body,
              source=str(src.relative_to(WORKING)),
              model=model, notes=notes)


# -------------------------------------------------------------------------
# 3. Intervention condition prompts
# -------------------------------------------------------------------------
intervention_files = [
    ('intervention', 'be_creative_vs_be_effective',
     'Be-creative vs. be-effective intervention',
     WORKING / 'experiments/cap_interventions/creative_vs_effective/run_generation.py',
     '15 frontier release models',
     'Prompt-suffix manipulation on the 5 CAP tasks. d_z = +1.40 length-residualized. Paper §3.6 / §4.6.'),
    ('intervention', 'reasoning_on_off',
     'Reasoning on/off intervention',
     WORKING / 'experiments/cap_interventions/reasoning_on_off/run_generation.py',
     '10 reasoning-capable release models',
     'Toggle reasoning per provider API. d_z = +0.34 length-residualized composite shift. Paper §3.6 / §4.6.'),
]
for cat, prompt_id, title, src, model, notes in intervention_files:
    if not src.exists():
        _emit(cat, prompt_id, title,
              '[Source script not found at extraction time.]',
              source=str(src.relative_to(WORKING)),
              model=model, notes=notes + ' (NOT FOUND)')
        continue
    # Both intervention scripts assemble their prompts inside a build_prompt()
    # function with conditional branches per (task, condition). Emit the
    # function source verbatim so readers see all 10 prompt variants
    # (5 tasks × 2 conditions) with their conditional logic intact.
    fn_src = _extract_function_source(src, 'build_prompt')
    if fn_src:
        _emit(cat, prompt_id, title, fn_src,
              source=str(src.relative_to(WORKING)),
              model=model,
              notes=notes + ' Prompts are assembled in `build_prompt(task, prompt_id, item, condition)`; '
                            'the function source is reproduced verbatim above.')
    else:
        _emit(cat, prompt_id, title,
              '[Could not locate build_prompt() function in source; see file directly.]',
              source=str(src.relative_to(WORKING)),
              model=model, notes=notes)


# -------------------------------------------------------------------------
# 4. Build INDEX.md
# -------------------------------------------------------------------------
files = sorted(OUT.rglob('*.md'))
files = [f for f in files if f.name != 'INDEX.md']
by_cat = defaultdict(list)
for f in files:
    cat = f.parent.name
    by_cat[cat].append(f)

lines = ['# AGC-Bench LLM-judge prompts — full extraction\n',
         'Every LLM-judge prompt referenced by the paper, extracted from the source modules. '
         'Per-benchmark scoring prompts come verbatim from the source papers; the AGC-authored '
         'prompts (DQ audit, domain classifier, AGC-Human fairness, MuCE judgment, be-creative '
         'and reasoning interventions) are released here for the first time.\n',
         '## Provenance\n',
         'These prompts are the canonical production prompts used to generate every released '
         'score. They are extracted programmatically by `scripts/extract_judge_prompts.py` from '
         'two sources of truth: the registry (`data/registry/registry_metrics.yaml`) and the '
         'per-benchmark annotator modules (`llm_judge/*_annotator.py`). The extraction is '
         'filtered by the `released_in_v1` flag in the registry, so only prompts tied to a '
         'benchmark in the released set appear here. Each file\'s header records the source '
         'module, constant name, and judge model. The scoring-prompt file count corresponds to '
         'the LLM-judge subset of the released set; the remaining benchmarks use formula-'
         'based or model-based metrics with no judge prompt.\n',
         '## Coverage\n',
         f'- Per-benchmark scoring prompts: {len(by_cat["scoring"])} files',
         f'- Data-quality / on-task audit: {len(by_cat["audit"])} file(s)',
         f'- Domain classification: {len(by_cat["classification"])} file(s)',
         f'- AGC-Human (fairness-aware + style-flip): {len(by_cat["agc_human"])} file(s)',
         f'- External validation (MuCE): {len(by_cat["validation"])} file(s)',
         f'- Interventions (be-creative, reasoning on/off): {len(by_cat["intervention"])} file(s)',
         f'- **Total extracted**: {len(files)} prompts\n']

for cat, fs in sorted(by_cat.items()):
    lines.append(f'## {cat}\n')
    for f in sorted(fs):
        rel = f.relative_to(OUT)
        title = f.stem.replace('_', ' ')
        lines.append(f'- [{title}]({rel})')
    lines.append('')

(OUT / 'INDEX.md').write_text('\n'.join(lines))

print(f'Extracted {len(files)} prompts:')
for cat, fs in sorted(by_cat.items()):
    print(f'  {cat:<15} {len(fs)}')
print(f'\nWrote {OUT / "INDEX.md"}')
print(f'Wrote {len(files)} per-prompt markdown files under {OUT}/')
