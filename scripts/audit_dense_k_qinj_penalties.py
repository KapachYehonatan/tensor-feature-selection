#!/usr/bin/env python3
"""Dense-K reporting for frozen training-selected methods; no new tuning."""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_posthoc_penalty_search as search
import numpy as np
import pandas as pd
from scipy import sparse

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/dense_k_qinj_audit"
DATASET = "QinJ_2012"
K_VALUES = tuple(range(1, 101))
FROZEN_METHODS = {
    "taxon_coverage": "taxon_coverage_batch_17",
    "effect_log_coverage": "effect_log_coverage_batch_22",
    "coverage_power": "coverage_power_batch_23",
}
METRICS = ["auroc", "precision", "recall", "f1"]


def check_standard_points(dense, previous):
    if dense.requested_k.duplicated().any() or previous.requested_k.duplicated().any():
        raise ValueError("Duplicate K results")
    if not np.isfinite(dense[METRICS]).all().all():
        raise ValueError("Nonfinite dense-K metrics")
    aligned = dense.merge(previous, on="requested_k", suffixes=("_new", "_old"), validate="one_to_one")
    if len(aligned) != len(previous):
        raise ValueError("Missing previously evaluated K")
    np.testing.assert_allclose(aligned[[f"{m}_new" for m in METRICS]],
                               aligned[[f"{m}_old" for m in METRICS]], rtol=0, atol=1e-10)


def main():
    root = search.cached_root(DATASET)
    sources = {family: PROJECT / "outputs/posthoc_penalty_search" / batch for family, batch in FROZEN_METHODS.items()}
    paths = [Path(__file__), Path(search.__file__), PROJECT / "scripts/run_qinj_redundancy_analysis.py",
             PROJECT / "feature_selection/run_feature_selection.py", root / "metrics.csv",
             PROJECT / "outputs/preprocessing" / DATASET / "samples.csv", root / "outer/split.npz"]
    for representation in ("join", "bacteria"):
        paths.extend(root / "outer" / f"{representation}_{suffix}" for suffix in ("train.npz", "test.npz", "features.csv.gz"))
    for family, base in sources.items():
        paths.extend([base / "result.json", base / DATASET / f"{family}_selected.csv",
                      base / DATASET / "locked_parameters.csv", base / DATASET / "holdout_metrics.csv"])
        if json.loads((base / "result.json").read_text()).get("status") != "complete":
            raise ValueError(f"Complete {base} before auditing")
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    signature = hashlib.sha256(repr((states, K_VALUES, FROZEN_METHODS)).encode()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = OUTPUT / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text())["signature"] != signature:
        raise ValueError("Changed audit inputs: preserve previous outputs and use a new directory")
    protocol.write_text(json.dumps(dict(signature=signature, source_states=states, requested_k=K_VALUES,
        selection="NONE: parameters/prefixes frozen by previous training CV; dense held-out K reporting only",
        graph_recomputation=False, holdout_status="adaptive/exploratory, not independent confirmation"), indent=2))
    train, test, table, ytrain, ytest, _, test_ids = search.load_space(DATASET, "outer")
    reference = pd.read_csv(root / "metrics.csv")
    bacteria_train = sparse.load_npz(root / "outer/bacteria_train.npz")
    bacteria_test = sparse.load_npz(root / "outer/bacteria_test.npz")
    bacterial_table = pd.read_csv(root / "outer/bacteria_features.csv.gz")
    if bacteria_train.shape != (len(ytrain), len(bacterial_table)) or bacteria_test.shape != (len(ytest), len(bacterial_table)):
        raise ValueError("Bacterial reference alignment failed")
    frames, predictions, settings = [], [], []
    jobs = [("baseline", train[:, :100], test[:, :100], reference[reference.method.eq("baseline")], 0),
            ("bacteria", bacteria_train, bacteria_test, reference[reference.method.eq("bacteria")], 0)]
    for family, base in sources.items():
        directory = base / DATASET
        selected = pd.read_csv(directory / f"{family}_selected.csv").sort_values("selection_rank")
        order = pd.Index(table.feature_id).get_indexer(selected.feature_id)
        if np.any(order < 0) or selected.feature_id.duplicated().any() or not np.array_equal(selected.selection_rank, np.arange(1, len(selected)+1)):
            raise ValueError(f"Unaligned frozen feature prefix: {family}")
        locked = pd.read_csv(directory / "locked_parameters.csv")
        setting = locked[locked.family.eq(family)]
        if len(setting) != 1:
            raise ValueError(f"Ambiguous locked parameter: {family}")
        settings.append(setting.assign(dataset=DATASET))
        previous = pd.read_csv(directory / "holdout_metrics.csv")
        jobs.append((family, train[:, order], test[:, order], previous[previous.family.eq(family)], setting.iloc[0].strength))
    for family, a, b, previous, strength in jobs:
        metrics, prediction = search.evaluate(a, b, ytrain, ytest, K_VALUES)
        check_standard_points(metrics, previous)
        frames.append(metrics.assign(family=family, strength=strength))
        predictions.append(prediction.assign(family=family, sample_index=test_ids[prediction.test_position]))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / "predictions.csv.gz", index=False)
    pd.concat(settings, ignore_index=True).to_csv(OUTPUT / "frozen_training_settings.csv", index=False)
    peaks = metrics.loc[metrics.groupby("family").auroc.idxmax()].copy()
    peaks["interpretation"] = "exploratory dense-K peak of frozen training settings; NOT model selection"
    peaks.to_csv(OUTPUT / "curve_peaks.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"baseline": "Join mean | BH", "bacteria": "Bacteria | BH reference",
              "taxon_coverage": "Join mean | additive coverage penalty",
              "effect_log_coverage": "Join mean | logarithmic coverage penalty",
              "coverage_power": "Join mean | proportional coverage penalty"}
    labels["catalogue_remaining_power"] = "Join mean | remaining-catalogue penalty"
    labels["coverage_uncertainty_power"] = "Join mean | uncertainty-shrunken coverage"
    labels["coverage_alignment_power"] = "Join mean | direction-aligned coverage"
    labels["coverage_snr_power"] = "Join mean | smooth signal-to-noise coverage"
    labels["bh_coverage_power"] = "Join mean | BH evidence × source coverage"
    labels["coverage_partition_power"] = "Join mean | training-subset consistency × coverage"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, upper in zip(axes, (20, 100)):
        for family, curve in metrics[metrics.requested_k.le(upper)].groupby("family"):
            axis.plot(curve.requested_k, curve.auroc, label=labels[family], marker=".", linewidth=2)
        axis.set(title=f"K=1…{upper}", xlabel="Requested K", ylabel="AUROC")
        axis.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("QinJ_2012 — dense-K audit of frozen training-selected parameters\nExploratory holdout curves; no parameter or operating K selected here")
    fig.tight_layout(rect=(0, .17, 1, .9))
    fig.savefig(OUTPUT / "dense_k_auc.png", dpi=200)
    plt.close(fig)
    if states != [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]:
        raise ValueError("Audit source data/code changed")
    bacteria_peak = float(peaks[peaks.family.eq("bacteria")].auroc.iloc[0])
    matches = peaks[peaks.family.isin(FROZEN_METHODS) & peaks.auroc.ge(bacteria_peak)].family.tolist()
    (OUTPUT / "result.json").write_text(json.dumps(dict(status="complete", source_caches_unchanged=True,
        graph_recomputation=False, qinj_candidates_matching_bacteria_peak=matches, goal_achieved=False), indent=2))
    print(peaks[["family", "strength", "requested_k", "effective_k", "auroc"]].to_string(index=False, float_format=lambda x:f"{x:.3f}"), flush=True)


if __name__ == "__main__":
    main()
