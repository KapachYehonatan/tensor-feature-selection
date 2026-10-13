#!/usr/bin/env python3
"""Joint-CV transfer of the promising source-complementarity penalty."""
from pathlib import Path
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint
import pandas as pd

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/size_source_transfer"
GRID={"size_plus_source":[0,.05,.10,.25]}


def main():
    original=pd.read_csv(PROJECT/"outputs/method_comparison/fine_k_real_metrics.csv")
    datasets=sorted(original.loc[original.source.eq('real'),'dataset'].unique())
    joint.OUTPUT,joint.GRID=OUTPUT,GRID
    joint.DATASETS=['QinJ_2012',*[d for d in datasets if d!='QinJ_2012']]
    joint.main()
    models=pd.read_csv(OUTPUT/'cv_selected_models.csv')
    peaks=pd.read_csv(OUTPUT/'curve_peaks.csv')
    rows=[]
    for dataset,table in models.groupby('dataset'):
        primary=table[table.is_primary].iloc[0]
        reference=table[table.family.eq('baseline')].iloc[0]
        p=peaks[peaks.dataset.eq(dataset)].set_index('family')
        chosen_peak=p.loc[primary.family]
        rows.append(dict(dataset=dataset,chosen_family=primary.family,parameter=float(primary.strength),
            cv_chosen_k=int(primary.requested_k),cv_chosen_auc=float(primary.auroc),
            baseline_cv_chosen_k=int(reference.requested_k),baseline_cv_chosen_auc=float(reference.auroc),
            chosen_model_delta=float(primary.auroc-reference.auroc),
            candidate_peak_auc=float(chosen_peak.peak_auc),candidate_peak_k=int(chosen_peak.peak_k),
            baseline_peak_auc=float(p.loc['baseline','peak_auc']),baseline_peak_k=int(p.loc['baseline','peak_k']),
            bacteria_peak_auc=float(p.loc['bacteria','peak_auc']),
            peak_delta=float(chosen_peak.peak_auc-p.loc['baseline','peak_auc'])))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTPUT/'peak_transfer_summary.csv',index=False)
    others=summary[summary.dataset.ne('QinJ_2012')]
    result=json.loads((OUTPUT/'result.json').read_text())
    result.update(criterion='Peak prediction over K, not uniform-K or curve-average improvement',
        peak_losses=others[others.peak_delta < -1e-12].to_dict('records'),
        cv_selected_model_losses=others[others.chosen_model_delta < -1e-12].to_dict('records'),
        mean_peak_transfer_delta=float(others.peak_delta.mean()),
        all_other_peaks_nondecreasing=bool((others.peak_delta >= -1e-12).all()),
        peak_status='Exploratory known-holdout maxima of training-locked settings; NOT training parameter selection',
        goal_achieved=False)
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
