#!/usr/bin/env python3
"""Ablate the subset-effect discount without changing representations or BH."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search
from scripts import run_joint_k_penalty_selection as joint
from scripts import run_guarded_size_penalty as bootstrap
from scripts.run_fold_consistent_partition_transfer import choose_fold_consistent_settings
from scripts.audit_dense_k_qinj_penalties import check_standard_points
import numpy as np
import pandas as pd

OUTPUT=PROJECT/'outputs/posthoc_penalty_search/partition_discount_ablation_v2'
SOURCES={'coverage_power':PROJECT/'outputs/posthoc_penalty_search/coverage_power_transfer',
         'coverage_partition_power':PROJECT/'outputs/posthoc_penalty_search/coverage_partition_transfer'}
POWERS=[0,.05,.10,.25,.50,1,2]
K_VALUES=tuple(range(1,101))
BOOTSTRAPS=2000
VARIANTS={'effect_only':('coverage_power',[0]), 'consistency_only':('coverage_partition_power',[0]),
          'coverage_only':('coverage_power',POWERS), 'both':('coverage_partition_power',POWERS)}
METRICS=['auroc','precision','recall','f1']


def choose_component_setting(cv,family,powers):
    """Reference-free, training-only nomination; raw and guarded decisions differ."""
    rows=cv[cv.family.eq('baseline')|(cv.family.eq(family)&cv.strength.isin(powers))].copy()
    _,locked,_=joint.choose_joint_settings(rows,{family:powers},reference_families=())
    setting=locked[locked.family.eq(family)].iloc[0]
    guard=choose_fold_consistent_settings(rows,{family:powers},reference_families=())[2]
    return dict(family=family,gamma=float(setting.strength),cv_k=int(setting.requested_k),
                cv_auc=float(setting.mean_cv_auc),guard_choice=guard)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protocols={family:json.loads((root/'protocol.json').read_text()) for family,root in SOURCES.items()}
    datasets=protocols['coverage_power']['datasets']
    if set(datasets)!=set(protocols['coverage_partition_power']['datasets']) or len(set(datasets))!=8:
        raise ValueError('Expected identical eight-dataset experiments')
    # Reuse independent training-fold profiles, not outer-training rankings.
    paths=[Path(__file__),Path(search.__file__),Path(joint.__file__),Path(bootstrap.__file__),
           PROJECT/'scripts/run_fold_consistent_partition_transfer.py',PROJECT/'scripts/run_qinj_redundancy_analysis.py',
           PROJECT/'feature_selection/run_feature_selection.py']
    data_states=[]
    for family,root in SOURCES.items():
        if json.loads((root/'result.json').read_text())['status']!='complete':raise ValueError('Incomplete source')
        if protocols[family]['grid']!={family:POWERS} or protocols[family]['k_values']!=list(joint.K_VALUES):
            raise ValueError('Different training search grid')
        for name,size,mtime in protocols[family]['source_states']:
            p=Path(name)
            if p.suffix=='.py':continue  # Later code additions do not rewrite historical inputs.
            if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime):raise ValueError(f'Changed source data: {p}')
            data_states.append((name,size,mtime))
        paths.extend([root/'protocol.json',root/'result.json'])
        for dataset in datasets:
            d=root/dataset
            paths.extend(d/name for name in ('inner_metrics.csv','outer_statistics.csv.gz','holdout_metrics.csv','locked_parameters.csv'))
    if {x[0]:x[1:] for x in protocols['coverage_power']['source_states'] if not x[0].endswith('.py')}!={x[0]:x[1:] for x in protocols['coverage_partition_power']['source_states'] if not x[0].endswith('.py')}:
        raise ValueError('The source experiments used different representations/splits')
    paths.extend(Path(name) for name,_,_ in data_states)
    paths=list(dict.fromkeys(paths));states=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    signature=hashlib.sha256(repr((states,VARIANTS,K_VALUES,BOOTSTRAPS)).encode()).hexdigest()
    manifest=OUTPUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:raise ValueError('Changed ablation: use a new directory')
    manifest.write_text(json.dumps(dict(signature=signature,source_states=states,variants=VARIANTS,powers=POWERS,
        k_values=K_VALUES,datasets=datasets,bootstrap_samples=BOOTSTRAPS,graph_recomputation=False,
        nomination='Each variant independently selects its parameter/K from existing identical three-fold CV; raw ranking reported even if guard rejects',
        matched_comparisons='Toggle Part 2 at BOTH training-selected coverage powers; compare the same K and classifier, without new outer-test tuning',
        bootstrap='Stratified paired patient bootstrap at training-selected anchors; conditional on fixed fitted models, not selection/search uncertainty',
        inference='Subset effects share data with full-data screen; heuristic scores, not independent tests or corrected p-values',
        holdout='Already extensively explored; all peaks descriptive, not independent evidence'),indent=2))
    settings,cv_frames={},[]
    for dataset in datasets:
        cvs={family:pd.read_csv(root/dataset/'inner_metrics.csv',float_precision='round_trip') for family,root in SOURCES.items()}
        control=['fold','family','strength','requested_k','effective_k',*METRICS]
        a=cvs['coverage_power'][cvs['coverage_power'].family.isin(['baseline','bacteria'])][control].sort_values(control[:4])
        b=cvs['coverage_partition_power'][cvs['coverage_partition_power'].family.isin(['baseline','bacteria'])][control].sort_values(control[:4])
        pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True),check_exact=False,atol=1e-12,rtol=0)
        settings[dataset]={variant:choose_component_setting(cvs[family],family,powers) for variant,(family,powers) in VARIANTS.items()}
        cv_frames.extend(cv.assign(dataset=dataset,source_family=family) for family,cv in cvs.items())
    (OUTPUT/'all_training_choices.json').write_text(json.dumps(settings,indent=2))
    pd.concat(cv_frames,ignore_index=True).to_csv(OUTPUT/'source_inner_metrics.csv',index=False)
    bootstrap.BOOTSTRAPS=BOOTSTRAPS
    frames,summary,matched,diagnostics=[],[],[],[]
    for dataset in datasets:
        train,test,table,ytrain,ytest,_,_=search.load_space(dataset,'outer')
        stats=pd.read_csv(SOURCES['coverage_partition_power']/dataset/'outer_statistics.csv.gz',float_precision='round_trip')
        coverage=pd.read_csv(SOURCES['coverage_power']/dataset/'outer_statistics.csv.gz',float_precision='round_trip').source_coverage
        np.testing.assert_allclose(stats.source_coverage,coverage,rtol=0,atol=1e-12)
        if len(stats)!=len(table):raise ValueError('Unaligned statistics')
        memo={}
        def evaluate(family,gamma):
            key=(family,gamma)
            if key not in memo:
                order,_=search.select_penalized(train,ytrain,table,stats,family,gamma)
                metric,prediction=search.evaluate(train[:,order],test[:,order],ytrain,ytest,K_VALUES)
                memo[key]=(metric,prediction,order)
            return memo[key]
        original=pd.read_csv(search.cached_root(dataset)/'metrics.csv')
        baseline,_,_=evaluate('baseline',0)
        check_standard_points(baseline,original[original.method.eq('baseline')])
        frames.append(baseline.assign(dataset=dataset,variant='bh_baseline',gamma=0))
        reference_auc=float(original[original.method.eq('bacteria')].auroc.max())
        for variant,setting in settings[dataset].items():
            metric,_,order=evaluate(setting['family'],setting['gamma'])
            frames.append(metric.assign(dataset=dataset,variant=variant,gamma=setting['gamma']))
            if variant in ('coverage_only','both'):
                prefix=pd.read_csv(SOURCES[setting['family']]/dataset/f'{setting["family"]}_selected.csv').feature_id.to_numpy()
                np.testing.assert_array_equal(table.feature_id.to_numpy()[order],prefix)
                old=pd.read_csv(SOURCES[setting['family']]/dataset/'holdout_metrics.csv')
                check_standard_points(metric,old[old.family.eq(setting['family'])])
            peak=metric.loc[metric.auroc.idxmax()];chosen=metric[metric.requested_k.eq(setting['cv_k'])].iloc[0]
            guard=setting['guard_choice'];guarded,_p,_o=evaluate(guard['family'],guard['strength'])
            guarded_peak=guarded.loc[guarded.auroc.idxmax()]
            summary.append(dict(dataset=dataset,variant=variant,gamma=setting['gamma'],cv_k=setting['cv_k'],
                cv_auc=setting['cv_auc'],holdout_at_cv_k=float(chosen.auroc),raw_peak_auc=float(peak.auroc),raw_peak_k=int(peak.requested_k),
                guarded_family=guard['family'],guard_accepted=guard['guard_accepted'],guarded_cv_k=guard['requested_k'],
                guarded_auc_at_cv_k=float(guarded[guarded.requested_k.eq(guard['requested_k'])].auroc.iloc[0]),
                guarded_peak_auc=float(guarded_peak.auroc),guarded_peak_k=int(guarded_peak.requested_k),bacteria_standard_peak=reference_auc))
        # Component comparisons: BOTH arms use identical gamma/K, with no guard.
        for anchor in ('coverage_only','both'):
            setting=settings[dataset][anchor];gamma,k=setting['gamma'],setting['cv_k']
            a,pa,ia=evaluate('coverage_power',gamma);b,pb,ib=evaluate('coverage_partition_power',gamma)
            for label,metric in (('without_part2',a),('with_part2',b)):
                frames.append(metric.assign(dataset=dataset,variant=f'matched_{anchor}_{label}',gamma=gamma))
            at_a=a[a.requested_k.eq(k)].iloc[0];at_b=b[b.requested_k.eq(k)].iloc[0]
            def predictions_at(prediction):return {k:prediction[prediction.requested_k.eq(k)].sort_values('test_position').probability.to_numpy()}
            interval=bootstrap.bootstrap_differences(predictions_at(pb),predictions_at(pa),ytest,[k])
            peak_a=a.loc[a.auroc.idxmax()];peak_b=b.loc[b.auroc.idxmax()]
            cv_a=pd.read_csv(SOURCES['coverage_power']/dataset/'inner_metrics.csv',float_precision='round_trip')
            cv_b=pd.read_csv(SOURCES['coverage_partition_power']/dataset/'inner_metrics.csv',float_precision='round_trip')
            def cv_point(cv,family):return cv[cv.family.eq(family)&cv.strength.eq(gamma)&cv.requested_k.eq(k)].sort_values('fold').auroc.to_numpy()
            differences=cv_point(cv_b,'coverage_partition_power')-cv_point(cv_a,'coverage_power')
            matched.append(dict(dataset=dataset,anchor=anchor,gamma=gamma,fixed_k=k,
                without_part2_auc=float(at_a.auroc),with_part2_auc=float(at_b.auroc),delta=float(at_b.auroc-at_a.auroc),
                paired_ci_lower=interval['lower'],paired_ci_upper=interval['upper'],
                cv_mean_delta=float(differences.mean()),cv_fold_deltas=json.dumps(differences.tolist()),
                without_part2_peak=float(peak_a.auroc),without_part2_peak_k=int(peak_a.requested_k),
                with_part2_peak=float(peak_b.auroc),with_part2_peak_k=int(peak_b.requested_k),
                descriptive_peak_delta=float(peak_b.auroc-peak_a.auroc),top10_overlap=len(set(ia[:10])&set(ib[:10]))))
        effect=np.abs(table.effect.to_numpy());robust=np.maximum(0,effect-stats.partition_instability.to_numpy())
        diagnostics.append(dict(dataset=dataset,candidates=len(table),zeroed_by_part2=int(np.sum((effect>0)&(robust==0))),
            median_retained_effect_fraction=float(np.median(np.divide(robust,effect,out=np.zeros_like(effect),where=effect>0)))))
        print(f'{dataset}: samples={len(ytrain)}/{len(ytest)}, classes={np.bincount(ytrain).tolist()}, candidates={len(table)}, ranking settings={len(memo)}, matched anchors verified',flush=True)
    curves=pd.concat(frames,ignore_index=True);summary=pd.DataFrame(summary);matched=pd.DataFrame(matched)
    curves.to_csv(OUTPUT/'metrics.csv',index=False);summary.to_csv(OUTPUT/'variant_summary.csv',index=False)
    matched.to_csv(OUTPUT/'matched_comparisons.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUTPUT/'discount_diagnostics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,anchor in zip(axes,('coverage_only','both')):
        gamma=settings['QinJ_2012'][anchor]['gamma']
        for label in ('without_part2','with_part2'):
            curve=curves[curves.dataset.eq('QinJ_2012')&curves.variant.eq(f'matched_{anchor}_{label}')]
            ax.plot(curve.requested_k,curve.auroc,label=label.replace('_',' '),linewidth=2)
        ax.axhline(summary[summary.dataset.eq('QinJ_2012')].bacteria_standard_peak.iloc[0],color='grey',linestyle=':',label='QinJ bacteria peak')
        ax.set(title=f'Identical coverage power γ={gamma:g}\nPower nominated by {anchor.replace("_"," ")} training CV',xlabel='Requested K',ylabel='Holdout AUROC')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('QinJ: controlled Part-2 ablation — same candidates, coverage, classifier and K\nAlready explored holdout; no independent statistical confirmation')
    fig.tight_layout();fig.savefig(OUTPUT/'qinj_matched_power_auc.png',dpi=200);plt.close(fig)
    if states!=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]:raise ValueError('Ablation sources changed')
    (OUTPUT/'result.json').write_text(json.dumps(dict(status='complete',source_caches_unchanged=True,graph_recomputation=False,
        independent_statistical_validation=False,comparison='Paired component ablation; raw scoring and CV guard reported separately'),indent=2))
    print('VARIANT SUMMARY\n'+summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)
    print('MATCHED COMPARISONS\n'+matched.to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)


if __name__=='__main__':main()
