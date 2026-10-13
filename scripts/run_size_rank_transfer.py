#!/usr/bin/env python3
"""Check the QinJ rank blend on every dataset with training-only weights."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_posthoc_penalty_search as search
import numpy as np
import pandas as pd

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/size_rank_transfer"
FAMILY = "bh_size_rank_blend"
GRID = {FAMILY: [0, .25, .50, .75, 1]}
DOMAINS = {"fine": tuple(range(1,11)), "primary": search.PRIMARY_K,
           "large": (20,35,50,60,75,85,100)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    search.OUTPUT = OUTPUT
    original = pd.read_csv(PROJECT/"outputs/method_comparison/fine_k_real_metrics.csv")
    datasets = sorted(original.loc[original.source.eq("real"), "dataset"].unique())
    paths = [Path(__file__), Path(search.__file__), PROJECT/"scripts/run_qinj_redundancy_analysis.py",
             PROJECT/"feature_selection/run_feature_selection.py"]
    for dataset in datasets:
        root = search.cached_root(dataset)
        paths.extend([PROJECT/"outputs/preprocessing"/dataset/"samples.csv", root/"metrics.csv"])
        for name in ("fold_0", "fold_1", "fold_2", "outer"):
            paths.extend(root/name/file for file in ("join_train.npz", "join_test.npz", "join_features.csv.gz", "split.npz"))
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    signature = hashlib.sha256(repr((GRID, DOMAINS, states)).encode()).hexdigest()
    protocol = dict(signature=signature, grid=GRID, domains=DOMAINS, graph_recomputation=False,
        source_states=states, classifier="unchanged balanced L2 logistic C=1",
        family_provenance="Exploratory size-rank family; QinJ batch-11 CV chose w=1; known holdouts",
        selection="each dataset training CV K5..10; lower weight on exact ties; zero is original BH",
        transfer_requirement="report all datasets and K; no holdout-based baseline fallback")
    file = OUTPUT/"protocol.json"
    if file.exists() and json.loads(file.read_text())["signature"] != signature:
        raise ValueError("Protocol changed; preserve this transfer and use a new output directory")
    file.write_text(json.dumps(protocol, indent=2))
    settings = {}
    status = []
    # Lock every dataset's decision before this batch evaluates any holdout.
    for dataset in ["QinJ_2012", *[d for d in datasets if d != "QinJ_2012"]]:
        settings[dataset] = search.run_training(dataset, GRID)
        status.append(dict(dataset=dataset, phase="training_locked"))
        pd.DataFrame(status).to_csv(OUTPUT/"status.csv", index=False)
    (OUTPUT/"all_training_choices.json").write_text(json.dumps(
        {d: choice for d, (_, choice) in settings.items()}, indent=2))
    results, summaries = [], []
    for row in status:
        dataset = row["dataset"]
        locked, choice = settings[dataset]
        metrics = search.run_holdout(dataset, locked, choice).assign(dataset=dataset)
        results.append(metrics)
        for domain, ks in DOMAINS.items():
            selected = metrics[metrics.requested_k.isin(ks)]
            mean = selected.groupby("family").auroc.mean()
            baseline = selected[selected.family.eq("baseline")].set_index("requested_k").auroc
            candidate = selected[selected.family.eq(FAMILY)].set_index("requested_k").auroc
            delta = candidate-baseline
            summaries.append(dict(dataset=dataset, domain=domain, weight=choice["strength"],
                candidate_auc=float(mean[FAMILY]), baseline_auc=float(mean.baseline),
                bacteria_auc=float(mean.bacteria), delta=float(delta.mean()),
                minimum_k_delta=float(delta.min()), below_baseline_k=json.dumps(delta.index[delta < -1e-12].tolist())))
        row["phase"] = "complete"
        pd.DataFrame(status).to_csv(OUTPUT/"status.csv", index=False)
        pd.concat(results, ignore_index=True).to_csv(OUTPUT/"all_metrics.csv", index=False)
        pd.DataFrame(summaries).to_csv(OUTPUT/"summary.csv", index=False)
    final = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    if states != final:
        raise ValueError("Source caches or protocol code changed during transfer")
    summary = pd.DataFrame(summaries)
    others = summary[summary.dataset.ne("QinJ_2012")]
    result = dict(status="complete", source_caches_unchanged=True, graph_recomputation=False,
        losses=others[others.delta < -1e-12].to_dict("records"),
        mean_transfer_deltas=others.groupby("domain").delta.mean().to_dict(),
        all_domain_means_nondecreasing=bool((others.delta >= -1e-12).all()),
        all_reported_k_nondecreasing=bool((others.minimum_k_delta >= -1e-12).all()),
        goal_achieved=False)
    (OUTPUT/"result.json").write_text(json.dumps(result, indent=2))
    print(summary.to_string(index=False, float_format=lambda x:f"{x:.3f}"), flush=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
