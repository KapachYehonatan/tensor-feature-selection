#!/usr/bin/env python3
"""Six-method real-data ablation from completed CV and holdout caches only."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.analyze_partition_discount_ablation import choose_component_setting, POWERS
from scripts.run_joint_k_penalty_selection import choose_joint_settings
from scripts.plot_aggregate_coverage_comparison import aggregate
from scripts.plot_best_current_qinj_comparison import GRIDS
from scripts.summarize_method_comparison import METRICS, data_limits
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEARCH = PROJECT / 'outputs/posthoc_penalty_search'
ABLATION = SEARCH / 'partition_discount_ablation_v2'
CURRENT = SEARCH / 'fold_consistent_partition_batch_37'
OUTPUT = PROJECT / 'outputs/current_penalty_ablation'
METHODS = {
    'bacteria': 'Bacteria-collapsed | BH',
    'baseline': 'Bernoulli-correlation-join mean | BH baseline',
    'current_penalty': 'Current penalty | subset discount × coverage + safeguard',
    'without_part2': 'Current penalty − part 2 | coverage + safeguard',
    'without_part3': 'Current penalty − part 3 | subset discount + safeguard',
    'without_part5': 'Current penalty − part 5 | subset discount × coverage, no safeguard',
}
COLORS = ['#F58518', '#4C78A8', '#111111', '#54A24B', '#B279A2', '#E45756']
COMPONENTS = {'current_penalty': ('coverage_partition_power', POWERS, 'both'),
              'without_part2': ('coverage_power', POWERS, 'coverage_only'),
              'without_part3': ('coverage_partition_power', [0], 'consistency_only')}


def training_choices(power_cv, partition_cv):
    """All ablations retain training-only tuning; only part 5 removes the gate."""
    _, locked, _ = choose_joint_settings(partition_cv[partition_cv.family.isin(['baseline', 'bacteria'])], {})
    choices = {row.family: dict(family=row.family, strength=float(row.strength),
               requested_k=int(row.requested_k), mean_cv_auc=float(row.mean_cv_auc))
               for row in locked.itertuples()}
    for method, (family, powers, _) in COMPONENTS.items():
        cv = power_cv if family == 'coverage_power' else partition_cv
        choices[method] = choose_component_setting(cv, family, powers)['guard_choice']
    rows = partition_cv[partition_cv.family.ne('bacteria')]
    choices['without_part5'] = choose_joint_settings(rows, {'coverage_partition_power': POWERS}, reference_families=())[2]
    return choices


def plot_metrics(data, title, path, ks, errors=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for (method, label), color in zip(METHODS.items(), COLORS):
            curve = data[data.method.eq(method)].sort_values('requested_k')
            ax.plot(curve.requested_k, curve[metric], label=label, color=color,
                    marker='o', markersize=4, linewidth=2, linestyle='--' if method.startswith('without') else '-')
            if errors is not None:
                sd = errors[errors.method.eq(method)].set_index('requested_k').loc[curve.requested_k, metric].to_numpy()
                ax.fill_between(curve.requested_k, np.clip(curve[metric]-sd, 0, 1),
                                np.clip(curve[metric]+sd, 0, 1), color=color, alpha=.07)
            elif metric == 'auroc':
                selected = curve[curve.requested_k.eq(curve.cv_selected_k)]
                ax.scatter(selected.requested_k, selected[metric], marker='*', s=160, color=color, edgecolor='white', zorder=5)
        ax.set(title=metric.upper(), xlabel='Requested K', ylabel=metric.upper(), xticks=ks, ylim=data_limits(data[metric]))
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, .13, 1, .92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    roots = [ABLATION, CURRENT, CURRENT/'dense_k_audit', SEARCH/'coverage_power_transfer', SEARCH/'coverage_partition_transfer']
    paths = [Path(__file__)]
    for root in roots:
        result = json.loads((root/'result.json').read_text())
        if result.get('status') != 'complete' or not result.get('source_caches_unchanged'):
            raise ValueError(f'Incomplete or modified-source cache: {root}')
        protocol = json.loads((root/'protocol.json').read_text())
        for name, size, mtime in protocol['source_states']:
            p = Path(name)
            if p.suffix != '.py' and (p.stat().st_size, p.stat().st_mtime_ns) != (size, mtime):
                raise ValueError(f'Changed historical data: {p}')
        paths.extend([root/'result.json', root/'protocol.json'])
    datasets = json.loads((ABLATION/'protocol.json').read_text())['datasets']
    if len(set(datasets)) != 8:
        raise ValueError('Expected eight independent-sample real datasets')
    paths.extend([ABLATION/'metrics.csv', ABLATION/'all_training_choices.json',
                  CURRENT/'all_training_choices.json', CURRENT/'dense_k_audit/metrics.csv',
                  SEARCH/'coverage_partition_transfer/all_training_choices.json'])
    paths.extend(SEARCH/root/dataset/'inner_metrics.csv'
                 for root in ('coverage_power_transfer', 'coverage_partition_transfer') for dataset in datasets)
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    signature = hashlib.sha256(repr((states, METHODS, COMPONENTS, GRIDS)).encode()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Changed analysis inputs; preserve existing outputs and choose a new directory')
    manifest.write_text(json.dumps(dict(signature=signature, source_states=states, datasets=datasets,
        methods=METHODS, grids=GRIDS, graph_recomputation=False, model_refitting=False,
        part2='B=max(0, min(abs(E), min_s sign(E)*E_s)); three disjoint stratified training subsets',
        part3='R=sum_t n_jt^2/(S_j*N_t); current score B*R^gamma; without part2 abs(E)*R^gamma; without part3 B',
        part5='Accept candidate only with positive paired mean CV gain and no fold loss versus BH at its own CV K',
        tuning='Each ablation independently tunes parameter/K on original isolated three-fold training CV; safeguard retained except without part5',
        without_part5='Ordinary joint mean-CV selection including BH fallback; only the additional fold-consistency gate is removed',
        aggregation='Equal dataset weights; shaded between-dataset SD is not a confidence interval',
        inference='Known holdouts: exploratory curves and maxima, not independent validation or new statistical tests'), indent=2))
    raw = pd.read_csv(ABLATION/'metrics.csv', float_precision='round_trip')
    audit = pd.read_csv(CURRENT/'dense_k_audit/metrics.csv', float_precision='round_trip')
    saved = json.loads((ABLATION/'all_training_choices.json').read_text())
    current = json.loads((CURRENT/'all_training_choices.json').read_text())
    unguarded = json.loads((SEARCH/'coverage_partition_transfer/all_training_choices.json').read_text())
    if any(set(s) != set(datasets) for s in (saved, current, unguarded)):
        raise ValueError('Different dataset sets')
    all_choices, frames, selections = {}, [], []
    for dataset in datasets:
        cvs = {family: pd.read_csv(SEARCH/root/dataset/'inner_metrics.csv', float_precision='round_trip')
               for family, root in [('power', 'coverage_power_transfer'), ('partition', 'coverage_partition_transfer')]}
        a = cvs['power'][cvs['power'].family.isin(['baseline', 'bacteria'])].sort_values(['family', 'requested_k', 'fold']).reset_index(drop=True)
        b = cvs['partition'][cvs['partition'].family.isin(['baseline', 'bacteria'])].sort_values(['family', 'requested_k', 'fold']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        choices = training_choices(cvs['power'], cvs['partition'])
        if choices['current_penalty'] != current[dataset] or choices['without_part5'] != unguarded[dataset]:
            raise ValueError(f'Original method training decision changed: {dataset}')
        all_choices[dataset] = choices
        for method, choice in choices.items():
            if method in COMPONENTS:
                family, powers, variant = COMPONENTS[method]
                setting = saved[dataset][variant]
                if choice != setting['guard_choice']:
                    raise ValueError('Component ablation guard changed')
            elif method == 'without_part5':
                variant = 'both'
                setting = saved[dataset][variant]
            if method == 'bacteria':
                curve = audit[audit.dataset.eq(dataset) & audit.family.eq('bacteria')].copy()
            elif choice['family'] == 'baseline':
                curve = raw[raw.dataset.eq(dataset) & raw.variant.eq('bh_baseline')].copy()
            else:
                if (choice['family'], choice['strength']) != (setting['family'], setting['gamma']):
                    raise ValueError('CV choice does not match cached raw curve')
                curve = raw[raw.dataset.eq(dataset) & raw.variant.eq(variant)].copy()
            curve = curve.sort_values('requested_k')
            if method in ('baseline', 'current_penalty'):
                ref = audit[audit.dataset.eq(dataset) & audit.family.eq('baseline' if method == 'baseline' else 'guarded_partition')].sort_values('requested_k')
                np.testing.assert_allclose(curve[['effective_k', *METRICS]], ref[['effective_k', *METRICS]], rtol=0, atol=1e-12)
            curve = curve[['requested_k', 'effective_k', *METRICS]].assign(dataset=dataset, method=method,
                underlying_family=choice['family'], parameter=choice['strength'], cv_selected_k=choice['requested_k'])
            frames.append(curve)
            point = curve[curve.requested_k.eq(choice['requested_k'])].iloc[0]
            selections.append(dict(dataset=dataset, method=method, **choice, holdout_auc_at_cv_k=float(point.auroc)))
        print(f'{dataset}: six cached curves; baseline/current decisions and metrics reproduced', flush=True)
    data = pd.concat(frames, ignore_index=True)
    aggregate(data, datasets, range(1, 101), methods=METHODS)  # Fail on missing, duplicate or nonfinite dense rows.
    data.to_csv(OUTPUT/'dense_metrics.csv', index=False)
    (OUTPUT/'all_training_choices.json').write_text(json.dumps(all_choices, indent=2))
    pd.DataFrame(selections).to_csv(OUTPUT/'training_choices.csv', index=False)
    summaries = []
    for domain, ks in {**GRIDS, 'fine_plus_coarse': tuple(sorted(set(GRIDS['fine']) | set(GRIDS['coarse'])))}.items():
        subset = data[data.requested_k.isin(ks)].copy()
        mean, sd = aggregate(subset, datasets, ks, methods=METHODS)
        peaks = subset.loc[subset.groupby(['dataset', 'method']).auroc.idxmax()].copy()
        refs = peaks[peaks.method.isin(['baseline', 'bacteria'])].pivot(index='dataset', columns='method', values='auroc')
        peaks['peak_delta_vs_baseline'] = peaks.auroc-peaks.dataset.map(refs.baseline)
        peaks['peak_delta_vs_bacteria'] = peaks.auroc-peaks.dataset.map(refs.bacteria)
        summaries.append(peaks.assign(grid=domain))
        if domain == 'fine_plus_coarse':
            continue
        subset.to_csv(OUTPUT/f'{domain}_k_metrics.csv', index=False)
        mean.to_csv(OUTPUT/f'{domain}_k_aggregate_mean.csv', index=False)
        sd.to_csv(OUTPUT/f'{domain}_k_aggregate_sd.csv', index=False)
        plot_metrics(mean, f'Current penalty ablation — all 8 real datasets ({domain} K)\nEqual-weight mean ± between-dataset SD; exploratory holdout', OUTPUT/f'{domain}_k_aggregate_real.png', ks, sd)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
        for ax, dataset in zip(axes.flat, datasets):
            local = subset[subset.dataset.eq(dataset)]
            plot_metrics(local, f'{dataset} — current penalty ablation ({domain} K)\nTraining-CV choices; ★ = CV-selected K when shown; exploratory holdout', OUTPUT/f'{domain}_k_{dataset}.png', ks)
            for (method, label), color in zip(METHODS.items(), COLORS):
                curve = local[local.method.eq(method)].sort_values('requested_k')
                ax.plot(curve.requested_k, curve.auroc, label=label, color=color, linewidth=2, linestyle='--' if method.startswith('without') else '-')
            ax.set(title=dataset, xlabel='Requested K', ylabel='AUROC', xticks=ks)
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
        fig.suptitle(f'Current penalty ablation — all datasets ({domain} K)\nFrozen training-only choices; exploratory holdout AUROC')
        fig.tight_layout(rect=(0, .13, 1, .93))
        fig.savefig(OUTPUT/f'{domain}_k_all_real_auc.png', dpi=180)
        plt.close(fig)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUTPUT/'peak_summary.csv', index=False)
    if states != [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]:
        raise ValueError('Source caches changed during comparison')
    (OUTPUT/'result.json').write_text(json.dumps(dict(status='complete', source_caches_unchanged=True,
        graph_recomputation=False, model_refitting=False, datasets=len(datasets), methods=len(METHODS), png_count=20), indent=2))
    print('BEST AUROC ON UNION OF FINE + COARSE GRIDS\n'+summary[summary.grid.eq('fine_plus_coarse')].pivot(index='dataset', columns='method', values='auroc').to_string(float_format=lambda x: f'{x:.3f}'), flush=True)


if __name__ == '__main__':
    main()
