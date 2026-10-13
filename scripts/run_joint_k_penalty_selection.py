#!/usr/bin/env python3
"""Joint training-CV choice of feature budget and a cached score penalty."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_posthoc_penalty_search as search
import numpy as np
import pandas as pd
from scipy import sparse

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/joint_k_batch_12"
GRID = {"bh_size_rank_blend": [0, .25, .50, .75, 1],
        "bh_size_rank_cap": [0, .25, .50, .75, 1]}
DATASETS = ["QinJ_2012"]
K_VALUES = search.K_VALUES
SELECTION_DESCRIPTION = "joint maximum mean three-fold CV AUC over parameter and K, including unchanged BH; smaller parameter then lower K then baseline on ties"


def choose_joint_settings(cv, grid, reference_families=("bacteria",)):
    """Pure CV choice; use reference_families=() for standalone deployment."""
    keys = ["family", "strength", "requested_k"]
    if cv.duplicated(["fold", *keys]).any() or not np.isfinite(cv.auroc).all():
        raise ValueError("Duplicate/nonfinite CV observations")
    families = {"baseline": [0], **{family: [0] for family in reference_families}, **grid}
    expected = {(family, weight, k) for family, weights in families.items()
                for weight in weights for k in K_VALUES}
    groups = cv.groupby(keys)
    if set(groups.groups) != expected or set(cv.fold) != {0,1,2} or not groups.size().eq(3).all():
        raise ValueError("Joint selection requires the complete three-fold parameter/K grid")
    mean = groups.agg(mean_cv_auc=("auroc", "mean"), cv_sd=("auroc", "std"),
                      mean_effective_k=("effective_k", "mean")).reset_index()
    locked = mean.sort_values(["family", "mean_cv_auc", "strength", "requested_k"],
        ascending=[True, False, True, True]).groupby("family").head(1).copy()
    family_order = {"baseline": -1, **{family: i for i, family in enumerate(grid)}}
    best = locked[locked.family.isin(family_order)].assign(family_order=lambda t:t.family.map(family_order)).sort_values(
        ["mean_cv_auc", "strength", "requested_k", "family_order"], ascending=[False, True, True, True]).iloc[0]
    choice = dict(family=best.family, strength=float(best.strength), requested_k=int(best.requested_k),
                  mean_cv_auc=float(best.mean_cv_auc))
    return mean, locked, choice


SELECTION_RULE = choose_joint_settings


def train(dataset, grid):
    directory = OUTPUT/dataset
    directory.mkdir(parents=True, exist_ok=True)
    root = search.cached_root(dataset)
    with np.load(root/"outer/split.npz") as split:
        outer_train, outer_test = split["train_index"], split["test_index"]
    frames = []
    for fold in range(3):
        name = f"fold_{fold}"
        train, test, table, ytrain, ytest, a, b = search.load_space(dataset, name)
        if search.SOURCE_FAMILIES.intersection(grid):table.attrs['source_weights']=search.source_weights(dataset,name,table)
        if search.CATALOGUE_FAMILIES.intersection(grid):table.attrs['catalogue_metadata']=search.catalogue_metadata(dataset,name,table)
        if np.intersect1d(np.r_[a,b], outer_test).size or set(np.r_[a,b]) != set(outer_train):
            raise ValueError("Training-fold isolation failed")
        stats = pd.DataFrame(dict(cluster_size=1/np.sqrt(table.pair_count.to_numpy(float))))
        if "coverage_support_power" in grid:
            stats["concentration"] = 1-search.effective_sample_support(train)
        if {"coverage_uncertainty_power", "coverage_snr_power"}.intersection(grid):
            stats["effect_lcb"] = search.penalty_statistics(train, ytrain, table).effect_lcb
        if "coverage_partition_power" in grid:
            stats["partition_instability"] = search.penalty_statistics(train, ytrain, table).partition_instability
        if "coverage_alignment_power" in grid:
            stats["linear_alignment"] = search.penalty_statistics(train, ytrain, table).linear_alignment
        if search.COVERAGE_FAMILIES.intersection(grid):
            stats=pd.concat([stats,search.coverage_statistics(dataset,name,table)],axis=1)
            stats.to_csv(directory/f'{name}_coverage.csv.gz',index=False)
        for c in sorted({c for family,cs in grid.items() if family in search.STABILITY_FAMILIES for c in cs}):
            stats[f"l1_frequency_{c:g}"]=search.resampled_l1_frequency(train,ytrain,c)
        if search.STABILITY_FAMILIES.intersection(grid):stats.to_csv(directory/f'{name}_stability.csv.gz',index=False)
        for family, weights in {"baseline": [0], **grid}.items():
            for weight in weights:
                order, _ = search.select_penalized(train, ytrain, table, stats, family, weight)
                scores, _ = search.evaluate(train[:,order], test[:,order], ytrain, ytest, K_VALUES)
                frames.append(scores.assign(fold=fold, family=family, strength=weight))
        bt = sparse.load_npz(root/name/"bacteria_train.npz")
        bv = sparse.load_npz(root/name/"bacteria_test.npz")
        bacterial_table = pd.read_csv(root/name/"bacteria_features.csv.gz")
        if bt.shape != (len(a),len(bacterial_table)) or bv.shape != (len(b),len(bacterial_table)):
            raise ValueError("Bacterial reference cache alignment failed")
        scores, _ = search.evaluate(bt,bv,ytrain,ytest,K_VALUES)
        frames.append(scores.assign(fold=fold,family="bacteria",strength=0))
        print(f"{dataset} {name}: samples={len(a)}/{len(b)}, classes={np.bincount(ytrain).tolist()}, "
              f"screened_join={len(table)}, screened_bacteria={len(bacterial_table)}",flush=True)
    cv = pd.concat(frames,ignore_index=True)
    cv.to_csv(directory/"inner_metrics.csv",index=False)
    mean, locked, choice = SELECTION_RULE(cv,grid)
    mean.to_csv(directory/"mean_cv_by_setting.csv",index=False)
    locked.to_csv(directory/"locked_parameters.csv",index=False)
    (directory/"training_choice.json").write_text(json.dumps(choice,indent=2))
    return locked,choice


def holdout(dataset, locked, choice):
    search.OUTPUT = OUTPUT
    # Bacterial predictors remain a separate reference, never a candidate input.
    metrics = search.run_holdout(dataset,locked[locked.family.ne("bacteria")],choice)
    selected, peaks = [], []
    for setting in locked.itertuples():
        curve = metrics[metrics.family.eq(setting.family)].sort_values("requested_k")
        selected_row = curve[curve.requested_k.eq(setting.requested_k)].iloc[0]
        selected.append(dict(dataset=dataset,family=setting.family,strength=float(setting.strength),
            requested_k=int(setting.requested_k),effective_k=int(selected_row.effective_k),
            mean_cv_auc=float(setting.mean_cv_auc),auroc=float(selected_row.auroc),
            precision=float(selected_row.precision),recall=float(selected_row.recall),f1=float(selected_row.f1),
            is_primary=setting.family==choice["family"]))
        peak = curve.loc[curve.auroc.idxmax()]
        peaks.append(dict(dataset=dataset,family=setting.family,strength=float(setting.strength),
            peak_k=int(peak.requested_k),peak_auc=float(peak.auroc),
            interpretation="exploratory holdout peak of training-locked weight; NOT model selection"))
    selected,peaks = pd.DataFrame(selected),pd.DataFrame(peaks)
    selected.to_csv(OUTPUT/dataset/"cv_selected_models.csv",index=False)
    peaks.to_csv(OUTPUT/dataset/"curve_peaks.csv",index=False)
    import matplotlib.pyplot as plt
    labels={"baseline":"Join mean | BH", "bacteria":"Bacterial means | BH reference",
            "bh_size_rank_blend":"Join mean | size-rank blend", "bh_size_rank_cap":"Join mean | size-rank cap",
            "size_plus_corr":"Join mean | size + correlation discounts",
            "size_plus_source":"Join mean | size + source-overlap discounts",
            "l1_stability":"Join mean | resampled-selection penalty",
            "size_l1_stability":"Join mean | size + resampled-selection penalties",
            "taxon_coverage":"Join mean | relative source-coverage penalty",
            "bh_log_coverage":"Join mean | BH evidence minus log-coverage penalty",
            "effect_log_coverage":"Join mean | MW effect minus log-coverage penalty",
            "coverage_power":"Join mean | proportional source-coverage penalty",
            "coverage_support_power":"Join mean | coverage × effective-support penalty",
            "bh_coverage_rank_blend":"Join mean | BH/coverage reliability rank blend",
            "catalogue_remaining_power":"Join mean | remaining-catalogue coverage penalty",
            "coverage_logistic_gain":"Join mean | coverage-weighted conditional logistic gain",
            "catalogue_completion_power":"Join mean | catalogue-completion coverage penalty",
            "coverage_uncertainty_power":"Join mean | uncertainty-shrunken coverage penalty",
            "coverage_alignment_power":"Join mean | direction-aligned coverage penalty",
            "coverage_snr_power":"Join mean | smooth signal-to-noise coverage penalty",
            "bh_coverage_power":"Join mean | BH evidence × source-coverage penalty",
            "coverage_partition_power":"Join mean | training-subset consistency × coverage",
            "size_taxon_coverage":"Join mean | size + relative-coverage penalties",
            "l1_importance":"Join mean | sparse-model importance penalty"}
    for name,ks in (("fine",range(1,11)),("coarse",[5,10,20,35,50,60,75,85,100])):
        fig,ax=plt.subplots(figsize=(11,6))
        for setting in locked.itertuples():
            curve=metrics[metrics.family.eq(setting.family)&metrics.requested_k.isin(ks)].sort_values("requested_k")
            label=f"{labels[setting.family]} (CV K={setting.requested_k}"
            if setting.family in search.RANK_FAMILIES:label+=f", w={setting.strength:.2f}"
            if setting.family=="size_plus_corr":label+=f", μ={setting.strength:.2f}"
            if setting.family=="size_plus_source":label+=f", μ={setting.strength:.2f}"
            if setting.family=="l1_importance":label+=f", scorer C={setting.strength:.2f}"
            if setting.family in search.STABILITY_FAMILIES:label+=f", scorer C={setting.strength:.2f}"
            if setting.family in search.CATALOGUE_FAMILIES:label+=f", γ={setting.strength:.2f}"
            if setting.family in search.COVERAGE_FAMILIES and setting.family not in search.RANK_FAMILIES:label+=f", {'γ' if setting.family in ('coverage_power','coverage_support_power','coverage_logistic_gain','coverage_uncertainty_power','coverage_alignment_power','coverage_snr_power','bh_coverage_power','coverage_partition_power') else 'λ'}={setting.strength:.2f}"
            line,=ax.plot(curve.requested_k,curve.auroc,marker="o",label=label+")")
            point=selected[selected.family.eq(setting.family)].iloc[0]
            if min(ks)<=point.requested_k<=max(ks):
                ax.scatter([point.requested_k],[point.auroc],s=140,marker="*",color=line.get_color(),zorder=5)
        ax.set(title=f"{dataset}: training-CV weight/K choices; stars mark chosen K\nExploratory holdout curves",
               xlabel="Requested K",ylabel="AUROC",xticks=ks)
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(OUTPUT/dataset/f"{name}_auc.png",dpi=200);plt.close(fig)
    print("CV-SELECTED MODELS\n"+selected.to_string(index=False,float_format=lambda x:f"{x:.3f}"),flush=True)
    print("EXPLORATORY CURVE PEAKS\n"+peaks.to_string(index=False,float_format=lambda x:f"{x:.3f}"),flush=True)
    return selected,peaks


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path(search.__file__),PROJECT/"scripts/run_qinj_redundancy_analysis.py",
           PROJECT/"feature_selection/run_feature_selection.py"]
    if SELECTION_RULE is not choose_joint_settings:
        paths.append(Path(sys.modules[SELECTION_RULE.__module__].__file__))
    for dataset in DATASETS:
        root=search.cached_root(dataset)
        paths.extend([PROJECT/"outputs/preprocessing"/dataset/"samples.csv",root/"metrics.csv"])
        metadata=bool(search.SOURCE_FAMILIES.intersection(GRID) or search.COVERAGE_FAMILIES.intersection(GRID) or search.CATALOGUE_FAMILIES.intersection(GRID))
        if metadata:paths.append(PROJECT/"outputs/preprocessing"/dataset/"edges.npz")
        for name in ("fold_0","fold_1","fold_2","outer"):
            if metadata:
                graph=PROJECT/"outputs/method_comparison/real"/dataset/"bernoulli_join_graph" if name=="outer" else root/name/"join"
                paths.append(graph/"assignments.npz")
            paths.append(root/name/"split.npz")
            for family in ("join","bacteria"):
                paths.extend(root/name/f"{family}_{suffix}" for suffix in ("train.npz","test.npz","features.csv.gz"))
    states=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    signature=hashlib.sha256(repr((GRID,K_VALUES,DATASETS,states)).encode()).hexdigest()
    file=OUTPUT/"protocol.json"
    if file.exists() and json.loads(file.read_text())["signature"]!=signature:
        raise ValueError("Changed protocol: preserve previous outputs and use a new directory")
    file.write_text(json.dumps(dict(signature=signature,source_states=states,grid=GRID,k_values=K_VALUES,
        datasets=DATASETS,graph_recomputation=False,classifier="unchanged balanced L2 logistic C=1",
        selection=SELECTION_DESCRIPTION,
        resampling=(dict(repeats=search.STABILITY_REPEATS,fraction=search.STABILITY_FRACTION,
            stratified=True,with_replacement=False,seed=search.SEED,coefficient_threshold=1e-8,
            status='training-only frequency heuristic, not a formal stability-selection error bound')
            if search.STABILITY_FAMILIES.intersection(GRID) else None),
        penalty_definitions={family: {
            "size_plus_corr":"abs(MW effect)-0.25/sqrt(pair count)-mu*max selected-only |Pearson r|",
            "l1_importance":"abs(MW effect)-0.25*(1-abs standardized L1 coefficient/max abs coefficient); strength is scorer C",
            "size_plus_source":"abs(MW effect)-0.25/sqrt(S)-mu*sum(taxon_fraction*max previously selected taxon_fraction); metadata only",
            "l1_stability":"abs(MW effect)-0.25*(1-resampled L1 selection frequency); strength is scorer C",
            "size_l1_stability":"abs(MW effect)-0.25/sqrt(S)-0.25*(1-resampled L1 selection frequency); strength is scorer C",
            "taxon_coverage":"abs(MW effect)-lambda*(1-sqrt(coverage)); coverage=sum_t n_jt^2/(S_j*N_t); counts only, no abundances",
            "bh_log_coverage":"-log10(cached MW p)-lambda*(-log10(coverage)); coverage=sum_t n_jt^2/(S_j*N_t); heuristic score, not a new p-value; zero restores BH order",
            "effect_log_coverage":"abs(cached MW effect)-lambda*(-log10(coverage)); same training count-only coverage; zero is effect order, not BH",
            "coverage_power":"abs(cached MW effect)*coverage^gamma; exported penalty is abs(effect)*(1-coverage^gamma), already discounted; zero is effect order, not BH",
            "coverage_support_power":"abs(cached MW effect)*(coverage*effective_support)^gamma; effective_support=(sum training sqrt-abundance)^2/(n*sum squared sqrt-abundance); exported penalty is actual effect loss, not multiplied again",
            "bh_coverage_rank_blend":"(1-w)*normalized BH rank+w*normalized rank(abs(MW effect)*coverage); coverage enters linearly, no imported gamma; zero is exactly BH; signed rank-disagreement penalty",
            "catalogue_remaining_power":"abs(effect)*(sum_t (n_jt/S_j)*(n_jt/N_t)*(1-covered_t))^gamma; covered_t=sum previously selected n_st/N_t; counts only; actual effect-loss penalty; zero is effect order",
            "coverage_logistic_gain":"min(cached MW normal-score squared, ridge-adjusted one-step training logistic gain conditional on selected)*coverage^gamma; actual evidence-loss penalty; balanced C=1 scorer; empty-set intercept-only null .5; heuristic, not calibrated significance",
            "catalogue_completion_power":"abs(effect)*(sum_t (n_jt/S_j)*(n_jt/N_t+covered_t))^gamma; counts partition original catalogue; coverage capped at 1; selected features support completing a shared source; actual effect-loss penalty",
            "coverage_uncertainty_power":f"max(0,abs(MW effect)-{search.UNCERTAINTY_Z}*training placement SE)*coverage^gamma; fixed uncertainty coefficient; heuristic not certified confidence; actual effect-loss export",
            "coverage_alignment_power":"min(abs(MW effect),max(0,sign(MW effect)*training Pearson(x,target)))*coverage^gamma; existing direction-misalignment statistic; actual effect loss; classifier unchanged",
            "coverage_snr_power":"abs(MW effect)*effect^2/(effect^2+training placement SE^2)*coverage^gamma; zero/zero scores zero; smooth reliability heuristic, not posterior or confidence; actual effect loss; zero power retains uncertainty discount",
            "bh_coverage_power":"-log10(cached MW p)*coverage^gamma; zero power restores original BH order among nonconstant candidates; actual evidence-loss export; heuristic score not corrected significance",
            "coverage_partition_power":"max(0,min(abs(cached MW effect),min direction-aligned MW effect across three stratified training subsets seed42))*coverage^gamma; existing partition-instability statistic; actual effect-loss export; not a confidence bound; zero power retains consistency discount",
            "size_taxon_coverage":"abs(MW effect)-0.25/sqrt(S)-lambda*(1-sqrt(coverage)); same count-only coverage definition",
        }.get(family,"fixed existing normalized size-rank adjustment") for family in GRID},
        references="BH join and bacteria each select their own K by the same CV criterion",
        fresh_dataset_application="compute training-only p-values and count coverage, use the same fixed grid/CV procedure; scoring has no dataset IDs, cached historical settings, or holdout input",
        criterion_update="User clarified that low-K success is valid despite high-K decline; curve averages are not success criteria",
        holdout_status="previously explored; peaks are descriptive only, never used to choose weights or K"),indent=2))
    settings={dataset:train(dataset,GRID) for dataset in DATASETS}
    (OUTPUT/"all_training_choices.json").write_text(json.dumps({d:c for d,(_,c) in settings.items()},indent=2))
    selected,peaks=[],[]
    for dataset,(locked,choice) in settings.items():
        a,b=holdout(dataset,locked,choice)
        selected.append(a);peaks.append(b)
    pd.concat(selected,ignore_index=True).to_csv(OUTPUT/"cv_selected_models.csv",index=False)
    pd.concat(peaks,ignore_index=True).to_csv(OUTPUT/"curve_peaks.csv",index=False)
    if states!=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]:
        raise ValueError("Source caches/code changed during joint selection")
    (OUTPUT/"result.json").write_text(json.dumps(dict(status="complete",source_caches_unchanged=True,
        graph_recomputation=False,goal_achieved=False),indent=2))


if __name__=="__main__":main()
