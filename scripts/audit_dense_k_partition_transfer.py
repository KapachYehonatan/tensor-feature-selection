#!/usr/bin/env python3
"""Audit frozen fold-consistent choices on a common dense K grid; no tuning."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search
from scripts.audit_dense_k_qinj_penalties import check_standard_points
from scripts.run_fold_consistent_partition_transfer import choose_fold_consistent_settings
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE=PROJECT/'outputs/posthoc_penalty_search/fold_consistent_partition_batch_37'
OUTPUT=SOURCE/'dense_k_audit'
K_VALUES=tuple(range(1,101))
FAMILY='guarded_partition'


def main():
    result=json.loads((SOURCE/'result.json').read_text())
    if result['status']!='complete' or not result['source_caches_unchanged']:
        raise ValueError('Complete the unchanged-source transfer first')
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    choices=json.loads((SOURCE/'all_training_choices.json').read_text())
    datasets=protocol['datasets']
    if len(set(datasets))!=8 or set(choices)!=set(datasets):raise ValueError('Expected all eight datasets')
    paths=[Path(__file__),Path(search.__file__),PROJECT/'scripts/run_qinj_redundancy_analysis.py',
           PROJECT/'feature_selection/run_feature_selection.py',PROJECT/'scripts/run_fold_consistent_partition_transfer.py',
           SOURCE/'protocol.json',SOURCE/'result.json',SOURCE/'all_training_choices.json']
    for dataset in datasets:
        root=search.cached_root(dataset);d=SOURCE/dataset;choice=choices[dataset]
        cv=pd.read_csv(d/'inner_metrics.csv',float_precision='round_trip')
        if choose_fold_consistent_settings(cv[cv.family.ne('bacteria')],protocol['grid'],reference_families=())[2]!=choice:
            raise ValueError('Reference-free training decision changed')
        paths.extend([d/'inner_metrics.csv',d/'locked_parameters.csv',d/f'{choice["family"]}_selected.csv',
                      d/'holdout_metrics.csv',root/'metrics.csv',root/'outer/split.npz',
                      PROJECT/'outputs/preprocessing'/dataset/'samples.csv'])
        for representation in ('join','bacteria'):
            paths.extend(root/'outer'/f'{representation}_{suffix}' for suffix in ('train.npz','test.npz','features.csv.gz'))
    states=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    signature=hashlib.sha256(repr((states,K_VALUES,choices)).encode()).hexdigest()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=OUTPUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:
        raise ValueError('Changed audit inputs; preserve outputs and use a new directory')
    manifest.write_text(json.dumps(dict(signature=signature,source_states=states,requested_k=K_VALUES,
        datasets=datasets,selection='NONE: all choices locked by previous training CV; common dense-grid reporting only',
        graph_recomputation=False,holdout_status='adaptive exploratory; no independent confirmation'),indent=2))
    frames,predictions=[],[]
    for dataset in datasets:
        root=search.cached_root(dataset);d=SOURCE/dataset;choice=choices[dataset]
        train,test,table,ytrain,ytest,_,test_ids=search.load_space(dataset,'outer')
        prefix=pd.read_csv(d/f'{choice["family"]}_selected.csv').sort_values('selection_rank')
        order=pd.Index(table.feature_id).get_indexer(prefix.feature_id)
        if np.any(order<0) or prefix.feature_id.duplicated().any() or not np.array_equal(prefix.selection_rank,np.arange(1,len(prefix)+1)):
            raise ValueError('Unaligned frozen prefix')
        bt=sparse.load_npz(root/'outer/bacteria_train.npz');bv=sparse.load_npz(root/'outer/bacteria_test.npz')
        bacterial_table=pd.read_csv(root/'outer/bacteria_features.csv.gz')
        if bt.shape!=(len(ytrain),len(bacterial_table)) or bv.shape!=(len(ytest),len(bacterial_table)):
            raise ValueError('Bacterial reference alignment failed')
        previous=pd.read_csv(d/'holdout_metrics.csv')
        original=pd.read_csv(root/'metrics.csv')
        jobs=[('baseline',train[:,:100],test[:,:100],'baseline'),('bacteria',bt,bv,'bacteria'),
              (FAMILY,train[:,order],test[:,order],choice['family'])]
        for label,a,b,source_family in jobs:
            if label==FAMILY and choice['family']=='baseline':
                metric,prediction=baseline_metric.copy(),baseline_prediction.copy()
            else:metric,prediction=search.evaluate(a,b,ytrain,ytest,K_VALUES)
            check_standard_points(metric,previous[previous.family.eq(source_family)])
            if label in ('baseline','bacteria'):
                check_standard_points(metric,original[original.method.eq(label)])
            if label=='baseline':baseline_metric,baseline_prediction=metric.copy(),prediction.copy()
            frames.append(metric.assign(dataset=dataset,family=label,underlying_family=source_family,
                strength=choice['strength'] if label==FAMILY else 0,cv_selected_k=choice['requested_k'] if label==FAMILY else np.nan))
            predictions.append(prediction.assign(dataset=dataset,family=label,sample_index=test_ids[prediction.test_position]))
        print(f'{dataset}: training={len(ytrain)}, holdout={len(ytest)}, screened_join={len(table)}, '
              f'bacteria={len(bacterial_table)}, frozen_family={choice["family"]}, K=1..100 verified',flush=True)
    metrics=pd.concat(frames,ignore_index=True)
    metrics.to_csv(OUTPUT/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUTPUT/'predictions.csv.gz',index=False)
    peaks=metrics.loc[metrics.groupby(['dataset','family']).auroc.idxmax()].copy()
    peaks.to_csv(OUTPUT/'curve_peaks.csv',index=False)
    rows=[]
    for dataset in datasets:
        p=peaks[peaks.dataset.eq(dataset)].set_index('family')
        rows.append(dict(dataset=dataset,chosen_family=choices[dataset]['family'],
            peak_k=int(p.loc[FAMILY,'requested_k']),peak_auc=float(p.loc[FAMILY,'auroc']),
            baseline_peak_k=int(p.loc['baseline','requested_k']),baseline_peak_auc=float(p.loc['baseline','auroc']),
            bacteria_peak_auc=float(p.loc['bacteria','auroc']),
            peak_delta=float(p.loc[FAMILY,'auroc']-p.loc['baseline','auroc'])))
    summary=pd.DataFrame(rows);summary.to_csv(OUTPUT/'peak_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(20,10),sharex=True)
    labels={'baseline':'Bernoulli join | BH','bacteria':'Bacteria | BH reference',FAMILY:'Bernoulli join | fold-consistent selection'}
    for ax,dataset in zip(axes.flat,datasets):
        for family in ('baseline','bacteria',FAMILY):
            curve=metrics[metrics.dataset.eq(dataset)&metrics.family.eq(family)]
            ax.plot(curve.requested_k,curve.auroc,label=labels[family],linewidth=2)
        ax.set(title=dataset,xlabel='Requested K',ylabel='AUROC');ax.grid(alpha=.2)
    handles,names=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=3)
    fig.suptitle('All eight datasets — frozen training-only choices, dense K=1..100\nExploratory holdout reporting; no parameter or operating K chosen here')
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(OUTPUT/'dense_k_all_real.png',dpi=200);plt.close(fig)
    if states!=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]:raise ValueError('Audit inputs changed')
    qinj=summary[summary.dataset.eq('QinJ_2012')].iloc[0]
    other=summary[summary.dataset.ne('QinJ_2012')]
    goal=bool(qinj.peak_auc>=qinj.bacteria_peak_auc and other.peak_delta.ge(-1e-12).all())
    (OUTPUT/'result.json').write_text(json.dumps(dict(status='complete',source_caches_unchanged=True,
        graph_recomputation=False,qinj_matches_bacteria=bool(qinj.peak_auc>=qinj.bacteria_peak_auc),
        all_other_peaks_nondecreasing=bool(other.peak_delta.ge(-1e-12).all()),
        goal_benchmark_criteria_met=goal,scope='exploratory peaks over every integer K=1..100 for frozen training choices'),indent=2))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)


if __name__=='__main__':main()
