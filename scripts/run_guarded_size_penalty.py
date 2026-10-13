#!/usr/bin/env python3
"""Train-only one-SE gate for the cached small-cluster score penalty."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/guarded_size_batch_6"
STRENGTHS=(0,.05,.1,.25,.5,1)
DOMAINS={"fine":tuple(range(1,11)),"large":tuple(k for k in search.K_VALUES if k>10)}
SE_MULTIPLIER=1.0
BOOTSTRAPS=500
REQUIRE_CONSISTENT_FOLDS=False
CV_CACHE=None


def choose_guard(cv, ks):
    rows=cv[cv.requested_k.isin(ks)]
    base=rows[rows.family.eq("baseline")]
    if len(base)!=3*len(ks) or base.duplicated(["fold","requested_k"]).any():
        raise ValueError("Incomplete or duplicated baseline CV points")
    baseline=base.groupby("fold").auroc.mean()
    if len(baseline)!=3 or not np.isfinite(baseline).all():
        raise ValueError("Guard requires three finite training-validation folds")
    candidates=[]
    for strength,group in rows[rows.family.eq("cluster_size")].groupby("strength"):
        if len(group)!=3*len(ks) or group.duplicated(["fold","requested_k"]).any():
            raise ValueError("Incomplete or duplicated candidate CV points")
        fold=group.groupby("fold").auroc.mean().reindex(baseline.index)
        if not np.isfinite(fold).all():raise ValueError("Nonfinite candidate CV score")
        difference=fold-baseline
        gain=float(difference.mean())
        se=float(difference.std(ddof=1)/np.sqrt(len(difference)))
        candidates.append(dict(strength=float(strength),mean_auc=float(fold.mean()),
            baseline_auc=float(baseline.mean()),gain=gain,se=se,fold_deltas=difference.tolist()))
    if not candidates:raise ValueError("No candidate strengths")
    best=sorted(candidates,key=lambda row:(-row["mean_auc"],row["strength"]))[0]
    best["apply_penalty"]=bool(best["gain"]>SE_MULTIPLIER*best["se"]+1e-12)
    if REQUIRE_CONSISTENT_FOLDS:
        best["apply_penalty"]=best["apply_penalty"] and min(best["fold_deltas"])>=-1e-12
    best["selected_family"]="cluster_size" if best["apply_penalty"] else "baseline"
    return best


def training_choices(dataset):
    directory=OUTPUT/dataset
    directory.mkdir(parents=True,exist_ok=True)
    if CV_CACHE is not None:
        cv=pd.read_csv(CV_CACHE/dataset/"inner_metrics.csv")
        cv.to_csv(directory/"inner_metrics.csv",index=False)
        choices={name:choose_guard(cv,ks) for name,ks in DOMAINS.items()}
        (directory/"training_choices.json").write_text(json.dumps(choices,indent=2))
        print("LOCKED FROM VERIFIED CV",dataset,json.dumps(choices),flush=True)
        return choices
    _,_,_,_,_,outer_train,outer_test=search.load_space(dataset,"outer")
    frames=[]
    for fold in range(3):
        train,test,table,ytrain,ytest,a,b=search.load_space(dataset,f"fold_{fold}")
        if np.intersect1d(np.r_[a,b],outer_test).size or set(np.r_[a,b])!=set(outer_train):
            raise ValueError("Fold isolation failed")
        statistics=pd.DataFrame({"cluster_size":1/np.sqrt(table.pair_count.to_numpy(float))})
        for family,strengths in (("baseline",[0]),("cluster_size",STRENGTHS)):
            for strength in strengths:
                order,_=search.select_penalized(train,ytrain,table,statistics,family,strength)
                metric,_=search.evaluate(train[:,order],test[:,order],ytrain,ytest,search.K_VALUES)
                frames.append(metric.assign(fold=fold,family=family,strength=strength))
        print(f"TRAIN {dataset} fold={fold}: samples={len(a)}/{len(b)}, classes={np.bincount(ytrain).tolist()}, "
              f"candidates={len(table)}, represented_pairs={int(table.pair_count.sum())}",flush=True)
    cv=pd.concat(frames,ignore_index=True)
    cv.to_csv(directory/"inner_metrics.csv",index=False)
    choices={name:choose_guard(cv,ks) for name,ks in DOMAINS.items()}
    (directory/"training_choices.json").write_text(json.dumps(choices,indent=2))
    print("LOCKED",dataset,json.dumps(choices),flush=True)
    return choices


def bootstrap_differences(prediction,reference,y,ks):
    rng=np.random.default_rng(search.SEED)
    pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
    counts1=rng.multinomial(len(pos),np.full(len(pos),1/len(pos)),size=BOOTSTRAPS)
    counts0=rng.multinomial(len(neg),np.full(len(neg),1/len(neg)),size=BOOTSTRAPS)
    differences=[]
    for k in ks:
        a=prediction[k];b=reference[k]
        def concordance(p):
            return (p[pos,None]>p[neg]).astype(float)+.5*(p[pos,None]==p[neg])
        matrix=concordance(a)-concordance(b)
        differences.append(np.einsum("bi,ij,bj->b",counts1,matrix,counts0,optimize=True)/(len(pos)*len(neg)))
    values=np.mean(differences,axis=0)
    return dict(lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)))


def holdout(dataset,choices):
    directory=OUTPUT/dataset
    train,test,table,ytrain,ytest,_,test_idx=search.load_space(dataset,"outer")
    statistics=pd.DataFrame({"cluster_size":1/np.sqrt(table.pair_count.to_numpy(float))})
    frames,predictions=[],[]
    for name,ks in DOMAINS.items():
        choice=choices[name]
        for label,family in (("baseline","baseline"),("guarded_size",choice["selected_family"]),
                             ("unguarded_size","cluster_size")):
            order,penalty=search.select_penalized(train,ytrain,table,statistics,family,choice["strength"])
            metric,prediction=search.evaluate(train[:,order],test[:,order],ytrain,ytest,ks)
            frames.append(metric.assign(method=label,domain=name,applied=family=="cluster_size",strength=choice["strength"]))
            predictions.append(prediction.assign(method=label,sample_index=test_idx[prediction.test_position]))
            if label=="guarded_size":
                selected=table.iloc[order].copy()
                selected["selection_rank"]=np.arange(1,len(order)+1)
                selected["score"]=(np.abs(selected.effect.to_numpy())-choice["strength"]*penalty
                    if family=="cluster_size" else -np.log10(selected.p.to_numpy()))
                selected.to_csv(directory/f"{name}_selected_features.csv",index=False)
    metrics=pd.concat(frames,ignore_index=True)
    prediction=pd.concat(predictions,ignore_index=True)
    reference=pd.read_csv(search.cached_root(dataset)/"metrics.csv")
    for field in ("auroc","precision","recall","f1","effective_k"):
        np.testing.assert_allclose(metrics[metrics.method.eq("baseline")].sort_values("requested_k")[field],
            reference[reference.method.eq("baseline")].sort_values("requested_k")[field],atol=1e-10)
    bacterial=reference[reference.method.eq("bacteria")].copy()
    bacterial["method"]="bacteria"
    metrics=pd.concat([metrics,bacterial],ignore_index=True).assign(dataset=dataset)
    metrics.to_csv(directory/"holdout_metrics.csv",index=False)
    prediction.to_csv(directory/"holdout_predictions.csv.gz",index=False)
    maps={method:{k:g.sort_values("test_position").probability.to_numpy() for k,g in
        prediction[prediction.method.eq(method)].groupby("requested_k")} for method in ("baseline","guarded_size")}
    uncertainty={name:bootstrap_differences(maps["guarded_size"],maps["baseline"],ytest,ks) for name,ks in DOMAINS.items()}
    (directory/"paired_bootstrap.json").write_text(json.dumps(uncertainty,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name,ks in (("fine",DOMAINS["fine"]),("coarse",(5,10,20,35,50,60,75,85,100))):
        fig,ax=plt.subplots(figsize=(10,6))
        for method,group in metrics[metrics.requested_k.isin(ks)].groupby("method"):
            group=group.sort_values("requested_k")
            ax.plot(group.requested_k,group.auroc,marker="o",label=method,
                    linestyle="--" if method=="unguarded_size" else "-")
        ax.set(title=f"{dataset}: training-only guarded penalty (exploratory)",xlabel="Requested K",ylabel="AUROC")
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(directory/f"{name}_auc.png",dpi=200);plt.close(fig)
    return metrics,uncertainty


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(PROJECT/"outputs/method_comparison/fine_k_real_metrics.csv")
    datasets=sorted(original[original.source.eq("real")].dataset.unique())
    protocol=dict(strengths=STRENGTHS,domains=DOMAINS,se_multiplier=SE_MULTIPLIER,
        require_consistent_folds=REQUIRE_CONSISTENT_FOLDS,
        cv_cache=str(CV_CACHE) if CV_CACHE is not None else None,
        gate="best candidate CV mean must exceed BH mean by > one paired-fold standard error",
        score="abs(cached MW effect) - strength/sqrt(original pair count) if accepted; otherwise original BH order",
        development="developed after earlier transfer results; adaptive exploratory, no independent confirmation",
        graph_recomputation=False,bootstrap_samples=BOOTSTRAPS,
        nested_ranking="within each domain; fine and large domain rankings can differ",
        choice_order="all dataset/domain choices locked before this batch's holdout evaluation")
    inputs=[]
    for dataset in datasets:
        root=search.cached_root(dataset)
        inputs.append(PROJECT/"outputs/preprocessing"/dataset/"samples.csv")
        for name in ("fold_0","fold_1","fold_2","outer"):
            inputs.extend(root/name/file for file in ("join_train.npz","join_test.npz","join_features.csv.gz","split.npz"))
    protocol["input_state"]=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in inputs]
    if CV_CACHE is not None:
        saved=json.loads((CV_CACHE/"protocol.json").read_text())
        expected=json.loads(json.dumps(protocol))
        for key in ("input_state","strengths","domains"):
            if saved[key]!=expected[key]:raise ValueError(f"Incompatible training CV cache: {key}")
        protocol["cv_source_signature"]=saved["source_signature"]
    protocol["source_signature"]=hashlib.sha256(Path(__file__).read_bytes()+Path(search.__file__).read_bytes()).hexdigest()
    file=OUTPUT/"protocol.json"
    if file.exists() and json.loads(file.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError("Changed protocol: use a new batch directory")
    file.write_text(json.dumps(protocol,indent=2))
    choices={dataset:training_choices(dataset) for dataset in datasets}
    (OUTPUT/"all_training_choices.json").write_text(json.dumps(choices,indent=2))
    results,summary=[],[]
    for dataset in datasets:
        metrics,uncertainty=holdout(dataset,choices[dataset]);results.append(metrics)
        for domain,ks in DOMAINS.items():
            means=metrics[metrics.requested_k.isin(ks)].groupby("method").auroc.mean()
            decision=dict(choices[dataset][domain])
            decision["baseline_cv_auc"]=decision.pop("baseline_auc")
            summary.append(dict(dataset=dataset,domain=domain,guarded_auc=float(means.guarded_size),
                baseline_auc=float(means.baseline),bacteria_auc=float(means.bacteria),
                delta=float(means.guarded_size-means.baseline),**decision,**uncertainty[domain]))
        pd.DataFrame(summary).to_csv(OUTPUT/"summary.csv",index=False)
        pd.concat(results,ignore_index=True).to_csv(OUTPUT/"all_metrics.csv",index=False)
        print("EVALUATED",dataset,flush=True)
    summary=pd.DataFrame(summary)
    losses=summary[(summary.dataset!="QinJ_2012") & (summary.delta < -1e-12)]
    if protocol["input_state"]!=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in inputs]:
        raise ValueError("Source representations changed during the experiment")
    (OUTPUT/"result.json").write_text(json.dumps(dict(status="complete",goal_achieved=False,
        observed_transfer_losses=json.loads(losses.to_json(orient="records"))),indent=2))
    print(summary[["dataset","domain","apply_penalty","strength","delta","lower","upper"]].round(3).to_string(index=False),flush=True)


if __name__=="__main__":main()
