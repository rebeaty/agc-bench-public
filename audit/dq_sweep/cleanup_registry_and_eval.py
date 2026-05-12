"""Maintenance utility for registry and eval-script cleanup.

1. registry_master.yaml: 157 entries, 79 with scenario_file refs to files
   that don't exist. Fix by adding a `released_in_v1` boolean to every
   entry and removing the scenario_file field from non-released entries
   (so no dangling refs). Curation-pipeline transparency preserved.

2. eval_scripts/: 173 .sh files, only 78 correspond to shipped scenarios.
   Move non-release scripts to eval_scripts/_archive/ so users see a
   clean directory matching the shipped scenarios. Also tighten the
   self-skip in 00_run_all.sh which currently skips only `run_all`, not
   `00_run_all` or `00_run_all_parallel`.

This is a historical curation helper, not part of the primary reproduction
entry points. The release artifacts already reflect its cleanup.
"""
from pathlib import Path
import shutil
import yaml

B = Path(__file__).resolve().parents[2]

# ---- Registry --------------------------------------------------------
reg_path = B / 'data/registry/registry_master.yaml'
reg = yaml.safe_load(reg_path.read_text())
shipped = {p.stem.replace('_scenario', '') for p in (B / 'scenarios').glob('*_scenario.py')}
print(f'Registry entries: {len(reg["datasets"])}    Shipped scenarios: {len(shipped)}')

n_changed = 0
n_dropped_scenario_file = 0
for benchmark_id, entry in reg['datasets'].items():
    in_release = benchmark_id in shipped
    if entry.get('released_in_v1') != in_release:
        entry['released_in_v1'] = in_release
        n_changed += 1
    if not in_release and 'scenario_file' in entry:
        # Drop dangling reference — file doesn't exist in this bundle
        entry.pop('scenario_file')
        n_dropped_scenario_file += 1

# Order keys consistently for diff readability
def _ordered(entry):
    front = ['display_name', 'released_in_v1', 'source_paper', 'source_repo',
             'input_modality', 'output_modality', 'scenario_file']
    rest = [k for k in entry if k not in front]
    return {k: entry[k] for k in front + rest if k in entry}

reg['datasets'] = {k: _ordered(v) for k, v in reg['datasets'].items()}

reg_bak = reg_path.with_suffix('.yaml.before_cleanup')
shutil.copy2(reg_path, reg_bak)
reg_path.write_text(yaml.dump(reg, sort_keys=False, default_flow_style=False))
print(f'  added released_in_v1 flag to {n_changed} entries')
print(f'  removed dangling scenario_file from {n_dropped_scenario_file} entries')
print(f'  backup: {reg_bak}')

# ---- Eval scripts ----------------------------------------------------
es = B / 'eval_scripts'
arch = es / '_archive'
arch.mkdir(exist_ok=True)
moved = 0
kept = 0
for sh in es.glob('*.sh'):
    name = sh.stem
    if name.startswith('00_') or name in {'_helm_run', 'run_all'}:
        kept += 1
        continue
    if name in shipped:
        kept += 1
        continue
    target = arch / sh.name
    shutil.move(str(sh), str(target))
    moved += 1
print(f'\neval_scripts: moved {moved} non-release scripts to _archive/, kept {kept}')

# Tighten 00_run_all.sh self-skip
ras = es / '00_run_all.sh'
content = ras.read_text()
old_skip = '    [ "$name" = "run_all" ] && continue'
new_skip = (
    '    case "$name" in\n'
    '        00_run_all|00_run_all_parallel|run_all|_helm_run) continue ;;\n'
    '    esac'
)
if old_skip in content:
    content = content.replace(old_skip, new_skip)
    ras.write_text(content)
    print('  tightened 00_run_all.sh self-skip pattern')
else:
    print('  WARN: did not find expected skip pattern in 00_run_all.sh')


# ---- Summary --------------------------------------------------------
post_check = sum(
    1 for benchmark_id, e in yaml.safe_load(reg_path.read_text())['datasets'].items()
    if e.get('scenario_file')
    and not (B / e['scenario_file']).exists()
)
print(f'\nDangling scenario_file refs after cleanup: {post_check}')
remaining = list((B / 'eval_scripts').glob('*.sh'))
remaining_non_release = [
    p for p in remaining
    if p.stem not in shipped
    and not p.stem.startswith('00_')
    and p.stem not in {'_helm_run', 'run_all'}
]
print(f'Non-release .sh files remaining in eval_scripts/: {len(remaining_non_release)}')
