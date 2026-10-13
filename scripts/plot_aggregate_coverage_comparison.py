#!/usr/bin/env python3
"""Add the two training-selected coverage protocols to real-data aggregates."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.plot_best_current_qinj_comparison import append_coverage, GRIDS, SEARCH
from scripts.plot_updated_qinj_comparison import (
    COMPARISON, METRICS, OLD_METHODS, NEW_METHODS, plot_comparison,
)

TABLES = COMPARISON / "tables/updated_methods"
FIGURES = COMPARISON / "figures/updated_methods"
PROTOCOLS = {
    "coverage_original_grid": ("taxon_coverage_transfer", "original grid"),
    "coverage_refined_grid": ("taxon_coverage_refined_transfer", "refined grid"),
}
EXTRA_METHODS = {
    method: f"Bernoulli-join mean | coverage penalty or BH (training CV; {label})"
    for method, (_, label) in PROTOCOLS.items()
}
METHODS = [*OLD_METHODS, *NEW_METHODS, *EXTRA_METHODS]
NAME = "aggregate_real_with_coverage"


def aggregate(data, datasets, ks, methods=None):
    """Require the same complete dataset/method/K grid, then weight equally."""
    keys = ["dataset", "method", "requested_k"]
    expected = pd.MultiIndex.from_product([datasets, METHODS if methods is None else methods, ks], names=keys)
    observed = pd.MultiIndex.from_frame(data[keys])
    if observed.has_duplicates or set(observed) != set(expected):
        raise ValueError("Missing or duplicate dataset/method/K rows in aggregate")
    if not np.isfinite(data[list(METRICS)]).all().all():
        raise ValueError("Nonfinite aggregate metrics")
    groups = data.groupby(["method", "requested_k"], as_index=False)
    mean = groups[list(METRICS)].mean()
    spread = groups[list(METRICS)].std()
    mean["n_datasets"] = len(datasets)
    mean["parameter"] = np.nan  # Different training-selected settings per dataset.
    return mean, spread


def main():
    marker = json.loads((PROJECT / "outputs/real_redundancy_analysis/complete.json").read_text())
    datasets = marker["datasets"]
    if marker.get("status") != "complete" or len(datasets) != 8 or len(set(datasets)) != 8:
        raise ValueError("The original eight-dataset comparison must be complete")
    choices = {}
    for method, (directory, _) in PROTOCOLS.items():
        root = SEARCH / directory
        result = json.loads((root / "result.json").read_text())
        if result.get("status") != "complete" or result.get("source_caches_unchanged") is not True:
            raise ValueError(f"Incomplete/modified-source protocol: {directory}")
        choices[method] = json.loads((root / "all_training_choices.json").read_text())
        if set(choices[method]) != set(datasets):
            raise ValueError(f"Different dataset set: {directory}")
    selection_rows = []
    for domain, ks in GRIDS.items():
        legacy_grid = "fine_k" if domain == "fine" else "coarse"
        original = pd.read_csv(TABLES / f"{legacy_grid}_metrics.csv")
        tables = []
        for dataset in datasets:
            data = original[original.dataset.eq(dataset)].copy()
            for method, (directory, label) in PROTOCOLS.items():
                root = SEARCH / directory / dataset
                metrics = pd.read_csv(root / "holdout_metrics.csv")
                locked = pd.read_csv(root / "locked_parameters.csv")
                choice = choices[method][dataset]
                selected = locked[locked.family.eq(choice["family"])]
                if len(selected) != 1 or selected.iloc[0].strength != choice["strength"] or selected.iloc[0].requested_k != choice["requested_k"]:
                    raise ValueError(f"Training-choice/lock mismatch: {method}/{dataset}")
                data, _ = append_coverage(data, metrics, locked, method, label, ks,
                                          dataset=dataset, family=choice["family"])
                if domain == "fine":
                    selection_rows.append(dict(dataset=dataset, method=method, **choice))
            tables.append(data)
        data = pd.concat(tables, ignore_index=True)
        mean, spread = aggregate(data, datasets, ks)
        # Reproduce the existing ten-method aggregate, not just its references.
        saved = pd.read_csv(TABLES / f"{legacy_grid}_aggregate_mean.csv")
        old = mean[mean.method.isin([*OLD_METHODS, *NEW_METHODS])]
        aligned = old.merge(saved, on=["method", "requested_k"], suffixes=("_new", "_old"), validate="one_to_one")
        if len(aligned) != 10 * len(ks) or not np.allclose(
            aligned[[f"{m}_new" for m in METRICS]], aligned[[f"{m}_old" for m in METRICS]], rtol=0, atol=1e-7):
            raise ValueError("Existing ten-method aggregate changed")
        stem = f"{domain}_k_{NAME}"
        plot_comparison(mean, f"All 8 real datasets — equal-weight mean ± between-dataset SD ({domain} K)\n"
                        "Coverage protocols: training-CV choices; BH retained on HMP, LiJ, ZhuF\n"
                        "Exploratory known-holdout curves; shaded SD is not a confidence interval",
                        FIGURES / f"{stem}.png", ks, errors=spread, extra_methods=EXTRA_METHODS)
        data.to_csv(TABLES / f"{stem}_metrics.csv", index=False)
        mean.to_csv(TABLES / f"{stem}_mean.csv", index=False)
        spread.to_csv(TABLES / f"{stem}_sd.csv", index=False)
        print(f"Saved 8 datasets × 12 methods × {len(ks)} K values: {FIGURES / f'{stem}.png'}")
    pd.DataFrame(selection_rows).to_csv(TABLES / f"{NAME}_training_choices.csv", index=False)


if __name__ == "__main__":
    main()
