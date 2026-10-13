#!/usr/bin/env python3
"""Refit cached ranked features at K=1,...,10 and draw fine-grid figures."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from research_utils import predict_model  # noqa: E402
from scripts.run_method_comparison import (  # noqa: E402
    HOLDOUT_SEED, MODEL_SEED, output_path, real_jobs, METHOD_LABELS,
)
from scripts.summarize_method_comparison import (  # noqa: E402
    METHOD_ORDER, METRICS, ROOT, TABLES, plot_curves,
)


K_VALUES = range(1, 11)


def labels(job):
    source = Path(job["input"])
    return pd.read_csv(source / "samples.csv")["y"].to_numpy(dtype=int)


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
                "method": method_path.name, "method_label": METHOD_LABELS[method_path.name],
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
    for job in real_jobs():
        rows.extend(evaluate_job(job))
        print(f'Fine-K predictions: {job["dataset"]}', flush=True)
    metrics = pd.DataFrame(rows)
    keys = ['dataset', 'method', 'requested_k']
    expected = {(job['dataset'], method, k) for job in real_jobs() for method in METHOD_ORDER for k in K_VALUES}
    if (metrics.duplicated(keys).any() or set(map(tuple, metrics[keys].to_numpy())) != expected
            or not np.isfinite(metrics[METRICS]).all().all()):
        raise ValueError('Incomplete or duplicate fine-K comparison')
    metrics.to_csv(ROOT/'fine_k_real_metrics.csv', index=False, float_format='%.8g')
    TABLES.mkdir(parents=True, exist_ok=True)
    for dataset, group in metrics.groupby('dataset'):
        group.to_csv(TABLES/f'fine_k_real_{dataset}.csv', index=False)
        plot_curves(group, f'{dataset}: 80/20 holdout (K=1,...,10)', f'fine_k_real_{dataset}')
    grouped = metrics.groupby(['method', 'requested_k'])[METRICS]
    mean, spread = grouped.mean().reset_index(), grouped.std().reset_index()
    mean.assign(n_datasets=metrics.dataset.nunique()).to_csv(TABLES/'fine_k_aggregate_real.csv', index=False)
    plot_curves(mean, 'Real datasets: equal-weight mean ± between-dataset SD (fine K)',
                'fine_k_aggregate_real', spread)
    print(f'Saved fine-grid three-method results under {ROOT}', flush=True)


if __name__ == '__main__':
    main()
