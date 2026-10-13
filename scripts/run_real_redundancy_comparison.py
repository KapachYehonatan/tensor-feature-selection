#!/usr/bin/env python3
"""Fit the five requested join variants and extend both real-data K grids."""
from pathlib import Path
import hashlib
import json
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
# Import the runner first: it bounds native numerical threads before numpy.
from scripts import run_qinj_redundancy_analysis as study
from scripts.plot_updated_qinj_comparison import (
    COMPARISON, NEW_METHODS, OLD_METHODS, METRICS, load_comparison, plot_comparison,
)
import pandas as pd

ROOT = PROJECT / "outputs/real_redundancy_analysis"
FIGURES = COMPARISON / "figures/updated_methods"
TABLES = COMPARISON / "tables/updated_methods"
GRIDS = {"fine_k": tuple(range(1, 11)), "coarse": (5,10,20,35,50,60,75,85,100)}
PARAMETERS = {method: study.GRID[method] for method in NEW_METHODS}
WORKERS = 8


def completed(directory):
    marker, protocol = directory / "complete.json", directory / "protocol.json"
    if not marker.exists() or not protocol.exists():
        return False
    saved, done = json.loads(protocol.read_text()), json.loads(marker.read_text())
    return (done.get("status") == "complete" and done.get("signature") == saved.get("signature")
            and saved.get("representation_signature") == study.cache_signatures()[1]
            and saved.get("seed") == study.SEED and saved.get("inner_folds") == 3
            and saved.get("threshold") == study.THRESHOLD
            and all(saved.get("grid", {}).get(method) == grid for method, grid in PARAMETERS.items()))


def main():
    for directory in (ROOT, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
    original = {
        "fine_k": pd.read_csv(COMPARISON / "fine_k_real_metrics.csv"),
        "coarse": pd.read_csv(COMPARISON / "metrics_by_k.csv"),
    }
    datasets = sorted(original["fine_k"].loc[original["fine_k"].source.eq("real"), "dataset"].unique())
    coarse_datasets = set(original["coarse"].loc[original["coarse"].source.eq("real"), "dataset"])
    if set(datasets) != coarse_datasets:
        raise ValueError("Fine and coarse original comparisons cover different datasets")
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + Path(study.__file__).read_bytes()).hexdigest()
    order = ["QinJ_2012", *[d for d in datasets if d != "QinJ_2012"]]
    status = [dict(dataset=d, status="pending") for d in order]
    collected = {grid: [] for grid in GRIDS}
    (ROOT / "complete.json").write_text(json.dumps(dict(status="running", datasets=datasets,
                                                        script_signature=fingerprint), indent=2))
    for row in status:
        dataset = row["dataset"]
        study.INPUT = PROJECT / "outputs/preprocessing" / dataset
        study.BASE = COMPARISON / "real" / dataset
        study.OUTPUT = PROJECT / "outputs/qinj_redundancy_analysis" if dataset == "QinJ_2012" else ROOT / dataset
        study.GRID, study.WORKERS = PARAMETERS, WORKERS
        study.COMMUNITY_MAX_EDGES = 800_000_000 if dataset == "ZhuF_2020" else 500_000_000
        integration = json.loads((study.BASE / "bernoulli_join_graph/integration.json").read_text())
        study.THRESHOLD = float(integration["effective_threshold"])
        row.update(status="running", threshold=study.THRESHOLD, output=str(study.OUTPUT))
        pd.DataFrame(status).to_csv(ROOT / "status.csv", index=False)
        print(f"Starting {dataset}: fixed phi={study.THRESHOLD:.3f}, "
              f"edge cap={study.COMMUNITY_MAX_EDGES:,}, output={study.OUTPUT}", flush=True)
        try:
            if completed(study.OUTPUT):
                print(f"Reusing completed {dataset} results", flush=True)
            else:
                study.main()
            side = pd.read_csv(study.OUTPUT / "metrics.csv")
            for grid, ks in GRIDS.items():
                data = load_comparison(original[grid], side, dataset, ks)
                plot_comparison(data, f"{dataset} — updated comparison ({grid.replace('_', ' ')} K)",
                                FIGURES / f"{grid}_{dataset}.png", ks)
                data.to_csv(TABLES / f"{grid}_{dataset}.csv", index=False)
                collected[grid].append(data)
            locked = pd.read_csv(study.OUTPUT / "locked_parameters.csv")
            locked[locked.method.isin(NEW_METHODS)].assign(dataset=dataset).to_csv(
                TABLES / f"locked_parameters_{dataset}.csv", index=False)
            row["status"] = "complete"
        except Exception as error:
            row.update(status="failed", error=str(error))
            traceback.print_exc()
            print(f"FAILED {dataset}; preserving its caches and continuing other datasets", flush=True)
        pd.DataFrame(status).to_csv(ROOT / "status.csv", index=False)
        for grid, tables in collected.items():
            if tables:
                pd.concat(tables, ignore_index=True).to_csv(TABLES / f"{grid}_metrics.csv", index=False)
    if any(row["status"] != "complete" for row in status):
        (ROOT / "complete.json").write_text(json.dumps(dict(status="failed", datasets=status,
                                                            script_signature=fingerprint), indent=2))
        raise RuntimeError("Some datasets failed; see status.csv. Full aggregate figures were not generated.")
    summaries = []
    for grid, tables in collected.items():
        data = pd.concat(tables, ignore_index=True)
        # First one row per dataset/method/K, then equal dataset weights.
        if data.duplicated(["dataset", "method", "requested_k"]).any():
            raise ValueError("Duplicate dataset weights in aggregate")
        keys = ["method", "requested_k"]
        mean = data.groupby(keys, as_index=False)[METRICS].mean()
        error = data.groupby(keys, as_index=False)[METRICS].std()
        mean["parameter"] = float("nan")  # Settings differ across datasets.
        plot_comparison(mean, f"All {len(datasets)} real datasets — equal-weight mean ± SD ({grid.replace('_', ' ')} K)",
                        FIGURES / f"{grid}_aggregate_real.png", GRIDS[grid], errors=error)
        mean.to_csv(TABLES / f"{grid}_aggregate_mean.csv", index=False)
        error.to_csv(TABLES / f"{grid}_aggregate_sd.csv", index=False)
        summaries.append(data.groupby(["dataset", "method"], as_index=False)[METRICS].mean().assign(grid=grid))
    pd.concat(summaries, ignore_index=True).to_csv(TABLES / "dataset_curve_means.csv", index=False)
    (ROOT / "complete.json").write_text(json.dumps(dict(status="complete", datasets=datasets,
        script_signature=fingerprint, methods=[*OLD_METHODS, *NEW_METHODS], grids=GRIDS), indent=2))
    print(f"COMPLETE: {len(datasets)} datasets, fine/coarse and aggregate PNGs in {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
