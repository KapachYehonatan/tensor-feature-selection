#!/usr/bin/env python3
"""Refit cached ranked features at K=1,...,10 and draw fine-grid figures."""

from pathlib import Path
import os
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from feature_selection.run_feature_selection import predict_model  # noqa: E402
from scripts.run_method_comparison import (  # noqa: E402
    HOLDOUT_SEED, MODEL_SEED, output_path, real_jobs, synthetic_jobs,
)
from scripts.summarize_method_comparison import (  # noqa: E402
    METHOD_ORDER, METRICS, ROOT, plot_curves, seed_average,
)


K_VALUES = range(1, 11)
SOURCE = os.environ.get("FINE_K_SOURCE", "all")
if SOURCE not in {"all", "real", "synthetic"}:
    raise ValueError("FINE_K_SOURCE must be all, real, or synthetic")


def labels(job):
    source = Path(job["input"])
    if job["source"] == "real":
        return pd.read_csv(source / "samples.csv")["y"].to_numpy(dtype=int)
    return np.load(source / "target.npy").astype(int)


def evaluate_job(job):
    y = labels(job)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=HOLDOUT_SEED)
    train_index, test_index = next(split.split(np.zeros(len(y)), y))
    y_train, y_test = y[train_index], y[test_index]
    rows = []
    for method_path in sorted(
        path for path in output_path(job).iterdir()
        if path.name in METHOD_ORDER and (path / "ranked_train.npz").exists()
    ):
        train = sparse.load_npz(method_path / "ranked_train.npz")
        test = sparse.load_npz(method_path / "ranked_test.npz")
        ranking = pd.read_csv(method_path / "ranked_features.csv.gz")
        coarse = pd.read_csv(method_path / "metrics.csv").iloc[0]
        for requested_k in K_VALUES:
            effective_k = min(requested_k, train.shape[1])
            probability = predict_model(
                train[:, :effective_k], y_train, test[:, :effective_k], MODEL_SEED
            )
            prediction = probability >= 0.5
            counts = ranking.iloc[:effective_k]["level"].value_counts()
            rows.append({
                "source": job["source"], "dataset": job["dataset"], "n": job["n"],
                "seed": job["seed"], "scenario": job["scenario"],
                "method": method_path.name, "method_label": coarse["method_label"],
                "requested_k": requested_k, "effective_k": effective_k,
                "bacteria_features": int(counts.get("bacterium", 0)),
                "gene_features": int(counts.get("gene", 0)),
                "pair_features": int(counts.get("pair", 0)),
                "pair_cluster_features": int(counts.get("pair_cluster", 0)),
                "auroc": roc_auc_score(y_test, probability),
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
            })
    return rows


def main():
    rows = []
    jobs = []
    if SOURCE in {"all", "real"}:
        jobs.extend(real_jobs())
    if SOURCE in {"all", "synthetic"}:
        jobs.extend(synthetic_jobs())
    for job in jobs:
        rows.extend(evaluate_job(job))
        print(f"Completed {job['source']} {job['dataset']} seed={job['seed']}", flush=True)
    metrics = pd.DataFrame(rows)
    output_name = "fine_k_metrics.csv" if SOURCE == "all" else f"fine_k_{SOURCE}_metrics.csv"
    metrics.to_csv(ROOT / output_name, index=False, float_format="%.8g")

    real = metrics[metrics["source"].eq("real")]
    synthetic_runs = metrics[metrics["source"].eq("synthetic")]
    synthetic = seed_average(metrics)
    for dataset, group in real.groupby("dataset"):
        plot_curves(group, f"Real dataset: {dataset} (K=1,...,10)", f"fine_k_real_{dataset}")
    for n, group in synthetic_runs.groupby("n"):
        keys = ["method", "method_label", "requested_k"]
        plot_curves(
            group.groupby(keys, as_index=False)[METRICS].mean(),
            f"Synthetic full_signals: n={int(n)} (K=1,...,10; seed mean ± SD)",
            f"fine_k_synthetic_n_{int(n)}",
            group.groupby(keys, as_index=False)[METRICS].std(),
        )
    for name, group, title in (
        ("real", real, "Real datasets: K=1,...,10; mean ± SD"),
        ("synthetic", synthetic, "Synthetic full_signals: K=1,...,10; mean ± SD"),
    ):
        if group.empty:
            continue
        keys = ["method", "method_label", "requested_k"]
        plot_curves(
            group.groupby(keys, as_index=False)[METRICS].mean(), title,
            f"fine_k_aggregate_{name}",
            group.groupby(keys, as_index=False)[METRICS].std(),
        )
    print(f"Saved fine-grid metrics and figures under {ROOT}", flush=True)


if __name__ == "__main__":
    main()
