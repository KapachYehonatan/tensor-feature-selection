#!/usr/bin/env python3
"""Extend the unchanged five-fold pilot to six cohorts; never refit QinJ/HMP."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_cv_evaluation as study
import numpy as np
import pandas as pd

PILOT = PROJECT/'outputs/cv_evaluation'
OUTPUT = PILOT/'remaining'
ALL_OUTPUT = PILOT/'all_real'
PILOT_DATASETS = ['HMP_2019_ibdmdb_unique', 'QinJ_2012']
DATASETS = ['JieZ_2017', 'LiJ_2017', 'NielsenHB_2014_unique', 'RubelMA_2020', 'WirbelJ_2018', 'ZhuF_2020']
THRESHOLDS = {'JieZ_2017': .80, 'LiJ_2017': .80, 'NielsenHB_2014_unique': .70,
              'RubelMA_2020': .70, 'WirbelJ_2018': .80, 'ZhuF_2020': .75}
ALL_THRESHOLDS = {'HMP_2019_ibdmdb_unique': .70, 'QinJ_2012': .80, **THRESHOLDS}
BASE_REPORT = study.report


def protected_states():
    files = sorted(p for d in PILOT_DATASETS for p in (PILOT/d).rglob('*') if p.is_file())
    files += sorted(p for p in PILOT.iterdir() if p.is_file())
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def verify_pilot():
    result = json.loads((PILOT/'result.json').read_text())
    protocol = json.loads((PILOT/'protocol.json').read_text())
    if (result['status'] != 'complete' or not result['source_caches_unchanged']
            or set(result['completed_datasets']) != set(PILOT_DATASETS) or result['signature'] != protocol['signature']
            or protocol['folds'] != study.FOLDS or protocol['seed'] != study.SEED
            or protocol['k_values'] != study.K_VALUES or protocol['tune_k'] or protocol['permanent_holdout']
            or protocol['methods'] != study.METHODS or protocol['bh_fdr'] != .10
            or protocol['prevalence'] != [.05, 5] or protocol['memory_limit_gib'] != study.MEMORY_LIMIT_GIB
            or protocol['thresholds'] != {d: ALL_THRESHOLDS[d] for d in PILOT_DATASETS}
            or protocol['model'] != 'Training-scaled balanced L2 C=1'
            or protocol['leiden'] != dict(resolution=1., seed=17729, iterations=-1, objective='modularity', weight='abs(phi)')):
        raise ValueError('Pilot is incomplete or incompatible with this extension')


def label_report(root, title, thresholds):
    path = root/'REPORT.md'
    content = path.read_text().replace('# QinJ/HMP: five-fold full-pipeline evaluation', '# '+title)
    cuts = ', '.join(f'{d} {t:.2f}' for d, t in thresholds.items())
    content = content.replace('Predeclared absolute-phi thresholds: QinJ .80, HMP .70;',
                              'Predeclared absolute-phi thresholds: '+cuts+';')
    content = content.replace('two-complete-dataset results', 'all-dataset results')
    content = content.replace('logs/cv_evaluation.*', 'logs/cv_evaluation_remaining.*')
    path.write_text(content)


def consolidate(remaining_complete):
    complete = [*PILOT_DATASETS, *[d for d in DATASETS if d in remaining_complete]]
    ALL_OUTPUT.mkdir(parents=True, exist_ok=True)
    # Symlink views provide one dataset browser without duplicating graphs or editing sources.
    for dataset in complete:
        source = (PILOT if dataset in PILOT_DATASETS else OUTPUT)/dataset
        view = ALL_OUTPUT/dataset
        if view.exists() or view.is_symlink():
            if not view.is_symlink() or view.resolve() != source.resolve():
                raise ValueError(f'Unexpected existing dataset view: {view}')
        else: view.symlink_to(source.resolve(), target_is_directory=True)
    original = study.OUTPUT, study.DATASETS, study.THRESHOLDS
    try:
        study.OUTPUT, study.DATASETS, study.THRESHOLDS = ALL_OUTPUT, [*PILOT_DATASETS, *DATASETS], ALL_THRESHOLDS
        BASE_REPORT(complete)
        label_report(ALL_OUTPUT, 'All real datasets: five-fold full-pipeline evaluation', ALL_THRESHOLDS)
    finally:
        study.OUTPUT, study.DATASETS, study.THRESHOLDS = original
    pilot_status = pd.read_csv(PILOT/'status.csv')
    remaining_status = pd.read_csv(OUTPUT/'status.csv') if (OUTPUT/'status.csv').exists() else pd.DataFrame(
        [dict(dataset=d, fold=f, status='pending') for d in DATASETS for f in range(study.FOLDS)])
    pd.concat([pilot_status, remaining_status], ignore_index=True).to_csv(ALL_OUTPUT/'status.csv', index=False)
    remaining_result = json.loads((OUTPUT/'result.json').read_text()) if (OUTPUT/'result.json').exists() else {}
    done = (len(complete) == len(PILOT_DATASETS)+len(DATASETS)
            and remaining_result.get('status') == 'complete' and remaining_result.get('source_caches_unchanged') is True)
    protocols = {str(root): json.loads((root/'protocol.json').read_text())['signature'] for root in (PILOT, OUTPUT)}
    metrics = pd.read_csv(ALL_OUTPUT/'mean_fold_metrics.csv')
    expected = {(d, m, k) for d in complete for m in study.METHODS for k in study.K_VALUES}
    keys = ['dataset', 'method', 'requested_k']
    if metrics.duplicated(keys).any() or set(map(tuple, metrics[keys].to_numpy())) != expected or not np.isfinite(metrics[study.METRICS]).all().all():
        raise ValueError('Combined CV grid incomplete or duplicated')
    study.write_json(ALL_OUTPUT/'result.json', dict(status='complete' if done else 'partial',
        completed_datasets=complete, source_protocols=protocols, pilot_refitted=False,
        remaining_live_status=str(OUTPUT/'status.csv'), note='Combined status/report update at dataset checkpoints.'))


def main():
    verify_pilot()
    protected = protected_states()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    extension = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        datasets=DATASETS, thresholds=THRESHOLDS, pilot_protocol=json.loads((PILOT/'protocol.json').read_text())['signature'],
        native_threads=1, join_workers=study.JOIN_WORKERS, leiden_workers=study.COMMUNITY_WORKERS,
        memory_limit_gib=study.MEMORY_LIMIT_GIB, edge_limit=study.EDGE_LIMIT, no_pilot_refit=True)
    marker = OUTPUT/'extension_protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != extension:
        raise ValueError('Changed extension inputs; preserve results and use a new output directory')
    study.write_json(marker, extension)
    original = study.OUTPUT, study.DATASETS, study.THRESHOLDS, study.report
    study.OUTPUT, study.DATASETS, study.THRESHOLDS = OUTPUT, DATASETS, THRESHOLDS
    def report(complete):
        BASE_REPORT(complete)
        label_report(OUTPUT, 'Remaining six datasets: five-fold full-pipeline evaluation', THRESHOLDS)
        consolidate(complete)
    def progress(*args, **kwargs):
        if args and args[0] == 'COMPLETE: two datasets, five full-pipeline folds each, three methods, all K':
            args = ('COMPLETE: six remaining datasets, five full-pipeline folds each, three methods, all K', *args[1:])
        print(*args, **kwargs)
    study.report, study.print = report, progress
    print('Six remaining cohorts; 30 new folds; completed QinJ/HMP caches are protected', flush=True)
    try:
        study.main()
    finally:
        if protected != protected_states(): raise RuntimeError('Completed pilot files changed unexpectedly')
        result = json.loads((OUTPUT/'result.json').read_text()) if (OUTPUT/'result.json').exists() else {}
        consolidate(result.get('completed_datasets', []))
        study.OUTPUT, study.DATASETS, study.THRESHOLDS, study.report = original
        del study.print
    print('COMPLETE: all eight datasets; combined fine/coarse CV figures in '+str(ALL_OUTPUT), flush=True)


if __name__ == '__main__': main()
