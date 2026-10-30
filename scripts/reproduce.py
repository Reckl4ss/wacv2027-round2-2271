"""Run documented analyses without writing into frozen input/reference directories."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'data/frozen'
REFERENCE = ROOT / 'results/reference'
ENGINE = ROOT / 'src/engines'
GROUPS = {'orientation': 'orientation_controls', 'dimension': 'dimension_sensitivity', 'sampling': 'sampling_robustness'}

def compare_csv(expected, actual, tolerance=1e-10):
    def read(path):
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)
    fields, left = read(expected)
    fields2, right = read(actual)
    if fields != fields2 or len(left) != len(right):
        raise AssertionError(f'CSV schema/row mismatch: {expected.name}')
    count, maximum = 0, 0.0
    for i, (a, b) in enumerate(zip(left, right)):
        for key in fields:
            if a[key] == b[key]:
                continue
            try:
                x, y = float(a[key]), float(b[key])
            except (ValueError, TypeError):
                raise AssertionError(f'{expected.name}, row {i}, {key}: {a[key]!r} != {b[key]!r}')
            if math.isnan(x) and math.isnan(y):
                continue
            if not math.isclose(x, y, rel_tol=0, abs_tol=tolerance):
                raise AssertionError(f'{expected.name}, row {i}, {key}: {x} != {y}')
            maximum = max(maximum, abs(x - y))
            count += 1
    return {'file': expected.name, 'rows': len(left), 'numeric_differences_within_tolerance': count, 'max_abs_error': maximum}

def execute(task, runs):
    start = time.monotonic()
    output = runs / task
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    if task in GROUPS:
        group = GROUPS[task]
        cmd = [sys.executable, str(ENGINE/group/'run.py'), '--input-dir', str(INPUTS), '--output-dir', str(output)]
        cmd += ['--resamples', '1000', '--permutation-draws', '2000'] if task == 'sampling' else ['--nboot', '2000']
    elif task == 'sampling-verify':
        cmd = [sys.executable, str(ENGINE/'sampling_robustness/verify.py'), '--input-dir', str(INPUTS),
               '--outputs', str(REFERENCE/'sampling_robustness'), '--input-manifest', str(ROOT/'data/provenance/sampling_robustness/input_checksums.sha256'),
               '--selected-output', str(output/'selected_ids.csv.gz'), '--report', str(output/'report.json')]
    elif task == 'external-verify':
        cmd = [sys.executable, str(ENGINE/'external_coverage/verify.py'), '--output', str(REFERENCE/'external_coverage'), '--report', str(output/'report.json')]
    elif task == 'recent-verify':
        ref = REFERENCE/'recent_proxies'
        cmd = [sys.executable, str(ENGINE/'recent_proxies/verify.py'), '--round1-results', str(INPUTS),
               '--raw', str(ref/'raw_scores.jsonl.gz'), '--architecture-scores', str(ref/'architecture_scores.jsonl.gz'),
               '--correlations', str(ref/'correlations.csv'), '--results', str(ref/'results.json'), '--output', str(output/'report.json')]
    elif task == 'capacity':
        stage = output/'work'
        (stage/'code').mkdir(parents=True, exist_ok=True)
        (stage/'results').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'src/analysis/zoo_analysis.py', stage/'code/zoo_analysis.py')
        for file in INPUTS.glob('*.pkl'):
            shutil.copy2(file, stage/'results'/file.name)
        # --no-base-t also skips loading the frozen full-population targets.
        # The supplied cache makes --full independent of a NATS installation.
        cmd = [sys.executable, str(stage/'code/zoo_analysis.py'), '--full']
    elif task == 'search-points':
        cmd = [sys.executable, str(ROOT/'scripts/replay_search_points.py'), '--output', str(output)]
    else:
        raise ValueError(task)
    env = dict(os.environ, PYTHONUTF8='1', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    display_cmd = ['python', *[x.replace(str(ROOT), '.') for x in cmd[1:]]]
    (output/'command.json').write_text(json.dumps(display_cmd, indent=2)+'\n')
    print(f'{task}: running', flush=True)
    with (output/'stdout.log').open('w', encoding='utf-8') as stdout, (output/'stderr.log').open('w', encoding='utf-8') as stderr:
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    result = {'task':task, 'status':'PASS' if process.returncode == 0 else 'FAIL', 'exit_code':process.returncode, 'seconds':time.monotonic()-start}
    if process.returncode == 0 and task in GROUPS:
        for expected in sorted((REFERENCE/GROUPS[task]).glob('*.csv')):
            actual = output/expected.name
            if not actual.exists():
                raise AssertionError(f'Missing regenerated CSV: {actual}')
            comparisons.append(compare_csv(expected, actual))
        result['reference_csv_comparisons'] = comparisons
    if process.returncode == 0 and task == 'capacity':
        from compare_capacity import compare
        result['reference_comparison'] = compare(REFERENCE/'core/zoo_analysis_results.pkl', output/'work/results/zoo_analysis_results.pkl')
        if result['reference_comparison']['status'] != 'PASS':
            result['status'] = 'FAIL'
        result['scope'] = 'Full cached-population analysis, including cached early targets; upstream measurements not rerun.'
    result['python'] = sys.version.split()[0]
    result['packages'] = {p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','matplotlib')}
    (output/'execution.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(process.returncode or 1)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=[*GROUPS,'sampling-verify','external-verify','recent-verify','capacity','search-points','all-cpu'])
    parser.add_argument('--runs',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    args.runs=args.runs.resolve()
    for protected in [INPUTS,REFERENCE,ENGINE,ROOT/'paper']:
        if args.runs == protected or protected in args.runs.parents:
            parser.error('The output directory must not be inside frozen data, reference results, engines, or paper.')
    from verify_bundle import verify
    verify()
    tasks=[*GROUPS,'sampling-verify','external-verify','recent-verify','search-points','capacity'] if args.task == 'all-cpu' else [args.task]
    for task in tasks:
        execute(task,args.runs)
