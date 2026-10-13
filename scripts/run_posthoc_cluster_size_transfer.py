#!/usr/bin/env python3
"""Test the exploratory QinJ small-cluster penalty on all cached datasets."""
from pathlib import Path
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search
import pandas as pd

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/cluster_size_transfer"
GRID={"cluster_size": [0,.05,.1,.25]}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    search.OUTPUT=OUTPUT
    original=pd.read_csv(PROJECT/"outputs/method_comparison/fine_k_real_metrics.csv")
    datasets=sorted(original[original.source.eq("real")].dataset.unique())
    (OUTPUT/"protocol.json").write_text(json.dumps(dict(
        family="cluster_size",formula="abs(cached MW effect) - lambda/sqrt(original pair count)",
        grid=GRID,primary_k=search.PRIMARY_K,graph_recomputation=False,
        family_provenance="Exploratory candidate identified after batch_1 holdout; NOT its global CV winner",
        choice="separate training-fold strength per dataset; unpenalized effect-only and BH controls",
        primary_transfer_requirement="report all datasets; do not switch to baseline based on holdout losses"),indent=2))
    results,summaries,status=[],[],[]
    for dataset in ["QinJ_2012",*[d for d in datasets if d!="QinJ_2012"]]:
        directory=OUTPUT/dataset
        if dataset=="QinJ_2012":
            source=PROJECT/"outputs/posthoc_penalty_search/batch_1/QinJ_2012"
            directory.mkdir(exist_ok=True)
            locked=pd.read_csv(source/"locked_parameters.csv")
            locked=locked[locked.family.isin(["baseline","cluster_size"])].copy()
            choice=dict(family="cluster_size",strength=float(locked[locked.family.eq("cluster_size")].strength.iloc[0]))
            (directory/"training_choice.json").write_text(json.dumps(choice,indent=2))
        else:
            locked,choice=search.run_training(dataset,GRID)
        # Fixed effect-only control; it is not a parameter chosen on the holdout.
        locked=pd.concat([locked,pd.DataFrame([dict(family="effect_only",strength=0)])],ignore_index=True)
        locked.to_csv(directory/"locked_parameters.csv",index=False)
        metrics=search.run_holdout(dataset,locked,choice).assign(dataset=dataset)
        results.append(metrics)
        for family,group in metrics.groupby("family"):
            summaries.append(dict(dataset=dataset,family=family,
                fine_auc=float(group[group.requested_k<=10].auroc.mean()),
                primary_auc=float(group[group.requested_k.isin(search.PRIMARY_K)].auroc.mean()),
                k10_auc=float(group[group.requested_k.eq(10)].auroc.iloc[0])))
        status.append(dict(dataset=dataset,status="complete",strength=choice["strength"]))
        pd.DataFrame(status).to_csv(OUTPUT/"status.csv",index=False)
        pd.concat(results,ignore_index=True).to_csv(OUTPUT/"all_metrics.csv",index=False)
        pd.DataFrame(summaries).to_csv(OUTPUT/"summary.csv",index=False)
    summary=pd.DataFrame(summaries)
    pivot=summary.pivot(index="dataset",columns="family",values="primary_auc")
    difference=pivot.cluster_size-pivot.baseline
    audit=dict(status="complete",graph_recomputation=False,
        transfer_losses={d:float(v) for d,v in difference.items() if d!="QinJ_2012" and v<0},
        mean_transfer_delta=float(difference.drop("QinJ_2012").mean()),goal_achieved=False)
    (OUTPUT/"result.json").write_text(json.dumps(audit,indent=2))
    print(pivot.round(3).to_string(),flush=True)
    print(json.dumps(audit),flush=True)


if __name__=="__main__":main()
