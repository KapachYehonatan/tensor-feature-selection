#!/usr/bin/env python3
"""Append the strongest completed QinJ penalty protocols; never refit or tune."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.plot_updated_qinj_comparison import (
    COMPARISON, SIDE, METRICS, load_comparison, plot_comparison,
)

SEARCH = PROJECT / "outputs/posthoc_penalty_search"
PROTOCOLS = {
    "coverage_original_grid": ("taxon_coverage_batch_17", "original grid"),
    "coverage_refined_grid": ("taxon_coverage_refined_transfer", "refined grid"),
}
GRIDS = {"fine": tuple(range(1, 11)), "coarse": (5, 10, 20, 35, 50, 60, 75, 85, 100)}
NAME = "real_QinJ_2012_best_current_penalty"


def append_coverage(data, metrics, locked, method, protocol_label, ks,
                    dataset="QinJ_2012", family="taxon_coverage"):
    """Check both references and the CV lock before appending one fixed curve."""
    if family not in ("baseline", "taxon_coverage", "coverage_partition_power"):
        raise ValueError("Coverage protocol must choose join BH or coverage in training CV")
    for checked_family, reference in (("baseline", "bernoulli_correlation_join"),
                                      ("bacteria", "bh_bacteria"), (family, None)):
        curve = metrics[metrics.family.eq(checked_family) & metrics.requested_k.isin(ks)].sort_values("requested_k")
        if len(curve) != len(ks) or set(curve.requested_k) != set(ks):
            raise ValueError(f"Incomplete or duplicate K results: {method}/{checked_family}")
        if not np.isfinite(curve[list(METRICS)]).all().all():
            raise ValueError(f"Nonfinite metrics: {method}/{checked_family}")
        if reference:
            original = data[data.method.eq(reference)].sort_values("requested_k")
            if not np.allclose(original[list(METRICS)], curve[list(METRICS)], rtol=0, atol=1e-7):
                raise ValueError(f"Reference mismatch: {method}/{checked_family}")
    setting = locked[locked.family.eq(family)]
    if len(setting) != 1:
        raise ValueError("Expected exactly one training-CV coverage setting")
    strength, cv_k = float(setting.iloc[0].strength), int(setting.iloc[0].requested_k)
    all_candidate = metrics[metrics.family.eq(family)]
    if not np.isfinite(strength) or not all_candidate.strength.eq(strength).all():
        raise ValueError("Holdout parameter differs from training-CV lock")
    if len(all_candidate[all_candidate.requested_k.eq(cv_k)]) != 1:
        raise ValueError("Missing or duplicate training-selected K")
    candidate = all_candidate[all_candidate.requested_k.isin(ks)].copy()
    selector = ({"taxon_coverage":"relative source-coverage penalty",
                 "coverage_partition_power":"training-subset consistency × source coverage"}
                .get(family,"BH without penalty (CV retained)"))
    symbol = "γ" if family == "coverage_partition_power" else "λ"
    label = (f"Bernoulli-join mean | {selector} "
             f"({protocol_label}; {symbol}={strength:.3f}; CV K={cv_k})")
    candidate["method"], candidate["method_label"] = method, label
    candidate["source"], candidate["dataset"] = "real", dataset
    candidate["parameter"], candidate["cv_selected_k"] = strength, cv_k
    candidate["result_origin"] = f"training-CV post-hoc penalty / {protocol_label}"
    return pd.concat([data, candidate], ignore_index=True), label


def main():
    complete = json.loads((SIDE / "complete.json").read_text())
    if complete.get("status") != "complete" or complete["signature"] != json.loads((SIDE / "protocol.json").read_text())["signature"]:
        raise ValueError("Original redundancy analysis is not complete/compatible")
    inputs = {}
    for method, (directory, label) in PROTOCOLS.items():
        root = SEARCH / directory
        result = json.loads((root / "result.json").read_text())
        if result.get("status") != "complete" or result.get("source_caches_unchanged") is not True:
            raise ValueError(f"Incomplete or modified-source protocol: {directory}")
        qinj = root / "QinJ_2012"
        inputs[method] = (pd.read_csv(qinj / "holdout_metrics.csv"),
                          pd.read_csv(qinj / "locked_parameters.csv"), label)
    for domain, ks in GRIDS.items():
        original_path = "fine_k_real_metrics.csv" if domain == "fine" else "metrics_by_k.csv"
        data = load_comparison(pd.read_csv(COMPARISON / original_path),
                               pd.read_csv(SIDE / "metrics.csv"), k_values=ks)
        extra = {}
        for method, (metrics, locked, label) in inputs.items():
            data, extra[method] = append_coverage(data, metrics, locked, method, label, ks)
        path = COMPARISON / "figures" / f"{domain}_k_{NAME}.png"
        plot_comparison(data, f"QinJ_2012 — current penalty comparison ({domain} K)\n"
                        "Exploratory known-holdout curves; ★ = training-selected K for coverage",
                        path, k_values=ks, extra_methods=extra)
        data.to_csv(COMPARISON / "tables" / f"{domain}_k_{NAME}.csv", index=False)
        print(f"Saved {data.method.nunique()} methods × {len(ks)} K values: {path}")


if __name__ == "__main__":
    main()
