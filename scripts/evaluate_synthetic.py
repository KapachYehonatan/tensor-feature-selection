#!/usr/bin/env python3
"""Evaluate co-clustering and sequential feature selection on synthetic data."""

from pathlib import Path
import json
import multiprocessing as mp
import os
import resource
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from feature_selection.run_feature_selection import (  # noqa: E402
    cluster_training_representations,
    run_holdout_selection,
)
from research_utils import (  # noqa: E402
    contextual_profiles,
    exhaustive_similarity_clusters,
    marginal_matrix,
    normalize_rows,
    similarity_clusters,
    similarity_coclustering,
    sqrt_sparse,
)
from scripts.generate_synthetic import (  # noqa: E402
    CONFIG_PATH,
    PRESET_NAME,
    load_configuration,
    synthetic_worker_count,
    tensor_to_csr,
)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def algorithm_kwargs(settings):
    return {
        "prevalence_fraction": float(settings["prevalence_fraction"]),
        "minimum_samples": int(settings["minimum_samples"]),
        "metric": settings["metric"],
        "similarity_threshold": float(settings["similarity_threshold"]),
        "jaccard_threshold": float(settings["jaccard_threshold"]),
        "max_cluster_rounds": int(settings["max_cluster_rounds"]),
    }


def selection_kwargs(settings):
    return {
        **algorithm_kwargs(settings),
        "screen_fdr": float(settings["screen_fdr"]),
        "test_fraction": float(settings["test_fraction"]),
        "inner_folds_count": int(settings["inner_folds"]),
        "holdout_seed": int(settings["holdout_seed"]),
        "model_seed": int(settings["model_seed"]),
    }


def random_split(n, test_fraction, seed):
    rng = np.random.default_rng([int(seed), int(n), 991])
    order = rng.permutation(n)
    n_test = int(np.ceil(test_fraction * n))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def clustering_metrics(scope, level, active, labels, truth, total):
    sizes = np.bincount(labels)
    return {
        "scope": scope,
        "level": level,
        "ari": float(adjusted_rand_score(truth[active], labels)),
        "nmi": float(normalized_mutual_info_score(truth[active], labels)),
        "active_coverage": float(len(active) / total),
        "recovered_clusters": int(len(sizes)),
        "singleton_fraction": float(np.mean(sizes[labels] == 1)),
    }


def selection_scores(level, selected, truth, tested, universe_size):
    selected, truth, tested = set(selected), set(truth), set(tested)
    true_positive = len(selected & truth)
    false_positive = len(selected - truth)
    false_negative = len(truth - selected)
    tested_negative = tested - truth
    true_negative = len(tested_negative - selected)
    truth_retained = len(truth & tested)
    precision = true_positive / len(selected) if selected else np.nan
    recall = true_positive / len(truth) if truth else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if true_positive and precision + recall
        else 0.0
    )
    return {
        "level": level,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive / len(tested_negative) if tested_negative else np.nan,
        "false_discovery_proportion": false_positive / len(selected) if selected else 0.0,
        "selected": len(selected),
        "truth": len(truth),
        "truth_retained": truth_retained,
        "truth_retention_rate": truth_retained / len(truth) if truth else np.nan,
        "tested": len(tested),
        "universe": int(universe_size),
    }


def stratified_subset(labels, maximum):
    selected = []
    unique = np.unique(labels)
    per_cluster = max(1, maximum // len(unique))
    for cluster in unique:
        selected.extend(np.flatnonzero(labels == cluster)[:per_cluster])
    if len(selected) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), selected, assume_unique=False)
        selected.extend(remaining[: maximum - len(selected)])
    return np.asarray(sorted(selected[:maximum]), dtype=int)


def ppjoin_check(name, profiles, settings):
    labels, diagnostics = similarity_clusters(
        profiles,
        float(settings["similarity_threshold"]),
        float(settings["jaccard_threshold"]),
        settings["metric"],
        return_diagnostics=True,
        collect_checked_pairs=True,
    )
    exhaustive_labels, qualifying = exhaustive_similarity_clusters(
        profiles,
        float(settings["similarity_threshold"]),
        float(settings["jaccard_threshold"]),
        settings["metric"],
    )
    checked = set(map(tuple, diagnostics.pop("checked_pairs")))
    qualifying = set(map(tuple, qualifying))
    recall = len(checked & qualifying) / len(qualifying) if qualifying else 1.0
    return {
        "level": name,
        "features": int(profiles.shape[1]),
        "candidate_recall": float(recall),
        "labels_equal": bool(np.array_equal(labels, exhaustive_labels)),
        "qualifying_pairs": len(qualifying),
        **diagnostics,
    }


def save_cluster_table(path, truth, raw_active, raw_labels, pipeline_active, pipeline_labels, selected, true_selected):
    size = len(truth)
    raw = np.full(size, -1, dtype=int)
    pipeline = np.full(size, -1, dtype=int)
    raw[raw_active] = raw_labels
    pipeline[pipeline_active] = pipeline_labels
    table = pd.DataFrame({
        "entity": np.arange(size),
        "true_cluster": truth,
        "raw_cluster": raw,
        "pipeline_cluster": pipeline,
        "raw_active": raw >= 0,
        "pipeline_active": pipeline >= 0,
        "true_signal": np.isin(np.arange(size), list(true_selected)),
        "selected": np.isin(np.arange(size), list(selected)),
    })
    table.to_csv(path, index=False, compression="gzip")


def truth_sets(dataset, q, bacteria_truth, gene_truth):
    signals = pd.read_csv(dataset / "signal_truth.csv.gz")
    bacteria_clusters = set(
        signals.loc[signals["level"] == "bacterium", "cluster"].dropna().astype(int)
    )
    gene_clusters = set(
        signals.loc[signals["level"] == "gene", "cluster"].dropna().astype(int)
    )
    bacteria = set(np.flatnonzero(np.isin(bacteria_truth, list(bacteria_clusters))))
    genes = set(np.flatnonzero(np.isin(gene_truth, list(gene_clusters))))
    pair_rows = signals[signals["level"] == "pair"]
    pairs = set(
        (pair_rows["bacterium"].astype(int) * q + pair_rows["gene"].astype(int)).tolist()
    )
    pair_effects = {
        int(row.bacterium) * q + int(row.gene): float(row.effect)
        for row in pair_rows.itertuples()
    }
    return bacteria, genes, pairs, pair_effects


def long_metric_rows(n, seed, scenario, category, rows):
    output = []
    for row in rows:
        identifiers = {
            key: row[key]
            for key in ("scope", "level", "mode")
            if key in row
        }
        for metric, value in row.items():
            if (
                metric in identifiers
                or isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
            ):
                continue
            output.append({
                "n": n,
                "seed": seed,
                "scenario": scenario,
                "category": category,
                "scope": identifiers.get("scope", ""),
                "level": identifiers.get("level", identifiers.get("mode", "")),
                "metric": metric,
                "value": value,
            })
    return output


def evaluate_dataset(dataset, output, settings):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    n, p, q = map(int, metadata["shape"])
    seed, scenario = int(metadata["seed"]), metadata["scenario"]
    matrix, edge_bacteria, edge_gene, _ = tensor_to_csr(dataset)
    bacteria_truth = pd.read_csv(dataset / "bacteria_truth.csv.gz")["true_cluster"].to_numpy()
    gene_truth = pd.read_csv(dataset / "gene_truth.csv.gz")["true_cluster"].to_numpy()
    true_bacteria, true_genes, true_pairs, pair_effects = truth_sets(
        dataset, q, bacteria_truth, gene_truth
    )
    target_path = dataset / "target.npy"
    target = np.load(target_path) if target_path.exists() else None
    pair_prevalence = np.asarray(matrix.getnnz(axis=0)).ravel() / n
    data_rows = [{
        "level": "pair",
        "overall_sparsity": float(1 - matrix.nnz / (n * p * q)),
        "mean_prevalence": float(pair_prevalence.mean()),
        "prevalence_sd": float(pair_prevalence.std()),
        "prevalence_min": float(pair_prevalence.min()),
        "prevalence_q05": float(np.quantile(pair_prevalence, 0.05)),
        "prevalence_median": float(np.median(pair_prevalence)),
        "prevalence_q95": float(np.quantile(pair_prevalence, 0.95)),
        "prevalence_max": float(pair_prevalence.max()),
        "zero_support_fraction": float(np.mean(pair_prevalence == 0)),
    }]
    oracle_rows = []
    if target is not None:
        with np.load(dataset / "target_model.npz") as model:
            probability = model["probability"]
        oracle_rows = [{
            "level": "oracle",
            "roc_auc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability)),
            "expected_positive_fraction": float(probability.mean()),
            "realized_positive_fraction": float(target.mean()),
        }]

    selection_result = None
    if target is not None:
        selection_result = run_holdout_selection(
            matrix,
            target,
            edge_bacteria,
            edge_gene,
            **selection_kwargs(settings),
            verbose=False,
        )
        train_index = selection_result["train_index"]
        pipeline_details = selection_result["details"]
    else:
        train_index, test_index = random_split(
            n, float(settings["test_fraction"]), seed
        )
        _, pipeline_details = cluster_training_representations(
            matrix[train_index],
            matrix[test_index],
            edge_bacteria,
            edge_gene,
            **algorithm_kwargs(settings),
            verbose=False,
        )

    raw_train = normalize_rows(matrix[train_index])
    raw_result = similarity_coclustering(
        raw_train,
        edge_bacteria,
        edge_gene,
        p,
        q,
        float(settings["similarity_threshold"]),
        float(settings["jaccard_threshold"]),
        settings["metric"],
        int(settings["max_cluster_rounds"]),
        return_diagnostics=True,
        verbose=False,
    )
    raw_bacteria, raw_b_labels, raw_genes, raw_g_labels, raw_diagnostics = raw_result

    clustering_rows = [
        clustering_metrics("raw", "bacterium", raw_bacteria, raw_b_labels, bacteria_truth, p),
        clustering_metrics("raw", "gene", raw_genes, raw_g_labels, gene_truth, q),
        clustering_metrics(
            "pipeline",
            "bacterium",
            pipeline_details["active_bacteria"],
            pipeline_details["bacteria_labels"],
            bacteria_truth,
            p,
        ),
        clustering_metrics(
            "pipeline",
            "gene",
            pipeline_details["active_genes"],
            pipeline_details["gene_labels"],
            gene_truth,
            q,
        ),
    ]

    bacteria_marginal = marginal_matrix(raw_train, edge_bacteria, p)
    bacteria_ppjoin = ppjoin_check(
        "bacterium", sqrt_sparse(bacteria_marginal), settings
    )
    gene_profiles = contextual_profiles(
        raw_train,
        edge_gene,
        raw_b_labels[edge_bacteria],
        q,
        int(raw_b_labels.max()) + 1,
    )
    gene_subset = stratified_subset(
        gene_truth, min(int(settings["ppjoin_subset_features"]), q)
    )
    gene_ppjoin = ppjoin_check(
        "gene", sqrt_sparse(gene_profiles[:, gene_subset]), settings
    )
    ppjoin_rows = [bacteria_ppjoin, gene_ppjoin]

    selected_bacteria = set()
    selected_genes = set()
    selected_pairs = set()
    selection_rows = []
    prediction_rows = []
    if selection_result is not None:
        selected_bacteria = set(map(int, selection_result["selected_bacteria"]))
        selected_genes = set(map(int, selection_result["selected_genes"]))
        selected_pairs = set(map(int, selection_result["selected_pairs"]))
        retained_pairs = set(np.flatnonzero(pipeline_details["pair_mask"]))
        selection_rows = [
            selection_scores(
                "bacterium",
                selected_bacteria,
                true_bacteria,
                pipeline_details["active_bacteria"],
                p,
            ),
            selection_scores(
                "gene",
                selected_genes,
                true_genes,
                pipeline_details["active_genes"],
                q,
            ),
            selection_scores(
                "pair", selected_pairs, true_pairs, retained_pairs, p * q
            ),
        ]
        prediction_rows = selection_result["summary"].to_dict("records")

    save_cluster_table(
        output / "bacteria_clusters.csv.gz",
        bacteria_truth,
        raw_bacteria,
        raw_b_labels,
        pipeline_details["active_bacteria"],
        pipeline_details["bacteria_labels"],
        selected_bacteria,
        true_bacteria,
    )
    save_cluster_table(
        output / "gene_clusters.csv.gz",
        gene_truth,
        raw_genes,
        raw_g_labels,
        pipeline_details["active_genes"],
        pipeline_details["gene_labels"],
        selected_genes,
        true_genes,
    )
    selected_feature_rows = [
        {
            "level": "bacterium",
            "entity": entity,
            "pair": pd.NA,
            "bacterium": entity,
            "gene": pd.NA,
            "true_signal": entity in true_bacteria,
            "selected": entity in selected_bacteria,
            "effect": np.nan,
        }
        for entity in sorted(true_bacteria | selected_bacteria)
    ]
    selected_feature_rows += [
        {
            "level": "gene",
            "entity": entity,
            "pair": pd.NA,
            "bacterium": pd.NA,
            "gene": entity,
            "true_signal": entity in true_genes,
            "selected": entity in selected_genes,
            "effect": np.nan,
        }
        for entity in sorted(true_genes | selected_genes)
    ]
    selected_feature_rows += [
        {
            "level": "pair",
            "entity": pd.NA,
            "pair": pair,
            "bacterium": pair // q,
            "gene": pair % q,
            "true_signal": pair in true_pairs,
            "selected": pair in selected_pairs,
            "effect": pair_effects.get(pair, 0.0),
        }
        for pair in sorted(true_pairs | selected_pairs)
    ]
    pd.DataFrame(
        selected_feature_rows,
        columns=[
            "level", "entity", "pair", "bacterium", "gene",
            "true_signal", "selected", "effect",
        ],
    ).to_csv(
        output / "selected_features.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(clustering_rows).to_csv(
        output / "clustering_metrics.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(selection_rows).to_csv(
        output / "selection_metrics.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(prediction_rows).to_csv(
        output / "prediction_metrics.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(data_rows).to_csv(
        output / "data_metrics.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(oracle_rows).to_csv(
        output / "oracle_metrics.csv", index=False, float_format="%.6f"
    )

    diagnostics = {
        "raw": raw_diagnostics,
        "pipeline": pipeline_details["cluster_diagnostics"],
        "ppjoin_subset": ppjoin_rows,
    }
    with (output / "ppjoin_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(diagnostics), handle, indent=2, sort_keys=True)
        handle.write("\n")

    runtime = perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    metric_rows = []
    metric_rows += long_metric_rows(n, seed, scenario, "clustering", clustering_rows)
    metric_rows += long_metric_rows(n, seed, scenario, "selection", selection_rows)
    metric_rows += long_metric_rows(n, seed, scenario, "prediction", prediction_rows)
    metric_rows += long_metric_rows(n, seed, scenario, "data", data_rows)
    metric_rows += long_metric_rows(n, seed, scenario, "oracle", oracle_rows)
    metric_rows += long_metric_rows(n, seed, scenario, "ppjoin", ppjoin_rows)
    comparison_rows = []
    for scope, stages in (
        ("raw", raw_diagnostics),
        ("pipeline", pipeline_details["cluster_diagnostics"]),
    ):
        for stage in stages:
            comparison_rows.append({"scope": scope, "level": stage["stage"], **stage})
    metric_rows += long_metric_rows(
        n, seed, scenario, "clustering_diagnostics", comparison_rows
    )
    metric_rows += long_metric_rows(n, seed, scenario, "resource", [{
        "level": "process", "runtime_seconds": runtime, "peak_rss_mb": peak_rss_mb
    }])
    pd.DataFrame(metric_rows).to_csv(output / "run_metrics.csv", index=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready({
            "n": n,
            "p": p,
            "q": q,
            "seed": seed,
            "scenario": scenario,
            "nnz": matrix.nnz,
            "runtime_seconds": runtime,
            "peak_rss_mb": peak_rss_mb,
            "clustering": clustering_rows,
            "selection": selection_rows,
            "prediction": prediction_rows,
            "data": data_rows,
            "oracle": oracle_rows,
        }), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"Evaluated n={n} seed={seed} {scenario}: "
        f"{runtime:.1f}s, peak RSS {peak_rss_mb:.1f} MB",
        flush=True,
    )


def evaluate_worker(dataset, output, settings):
    evaluate_dataset(Path(dataset), Path(output), settings)


def markdown_table(frame):
    if frame.empty:
        return "No results."
    values = frame.copy()
    for column in values.select_dtypes(include="number"):
        values[column] = values[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.to_numpy()]
    return "\n".join([header, rule, *rows])


def aggregate_report(root, preset_name):
    files = sorted(root.glob("n_*/seed_*/*/run_metrics.csv"))
    if not files:
        raise ValueError("No synthetic evaluation results were produced")
    runs = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    runs["scope"] = runs["scope"].fillna("")
    runs.to_csv(root / "runs.csv", index=False, float_format="%.6f")
    summary = (
        runs.groupby(
            ["n", "scenario", "category", "scope", "level", "metric"],
            dropna=False,
        )["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(root / "summary.csv", index=False, float_format="%.6f")
    important = summary[
        (
            summary["category"].eq("clustering")
            & summary["scope"].eq("pipeline")
            & summary["metric"].isin(["ari", "nmi"])
        )
        | (
            summary["category"].eq("selection")
            & summary["metric"].isin([
                "precision", "recall", "false_positive", "truth_retention_rate"
            ])
        )
        | (
            summary["category"].eq("prediction")
            & summary["level"].isin([
                "marginal_clusters", "pairs_only", "marginal_plus_pairs"
            ])
            & summary["metric"].isin(["roc_auc", "average_precision"])
        )
        | (
            summary["category"].eq("ppjoin")
            & summary["metric"].eq("candidate_recall")
        )
        | (
            summary["category"].eq("data")
            & summary["metric"].isin([
                "overall_sparsity", "prevalence_q05", "prevalence_q95"
            ])
        )
        | (
            summary["category"].eq("oracle")
            & summary["metric"].isin([
                "roc_auc", "average_precision", "realized_positive_fraction"
            ])
        )
        | summary["category"].eq("resource")
    ]
    report = (
        f"# Synthetic evaluation: {preset_name}\n\n"
        "Key results are grouped by sample size, scenario, evaluation scope and feature "
        "level. Full numeric summaries are in `summary.csv`.\n\n"
        + markdown_table(important)
        + "\n"
    )
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print("\nSynthetic evaluation summary")
    for n in sorted(summary["n"].unique()):
        current = summary[summary["n"].eq(n)]

        def value(scenario, category, metric, level=None):
            rows = current[
                current["scenario"].eq(scenario)
                & current["category"].eq(category)
                & current["metric"].eq(metric)
            ]
            if level is not None:
                rows = rows[rows["level"].eq(level)]
            return float(rows["mean"].mean()) if len(rows) else np.nan

        exact_ari = value("exact_blocks", "clustering", "ari")
        ppjoin_recall = current.loc[
            current["category"].eq("ppjoin")
            & current["metric"].eq("candidate_recall"),
            "mean",
        ].min()
        null_false_positives = value(
            "null_target", "selection", "false_positive"
        )
        full_pair_recall = value("full_signals", "selection", "recall", "pair")
        pair_only_recall = value("pair_signals", "selection", "recall", "pair")
        sparsity = value("noisy_blocks", "data", "overall_sparsity", "pair")
        pair_retention = value(
            "full_signals", "selection", "truth_retention_rate", "pair"
        )
        max_runtime = current.loc[
            current["metric"].eq("runtime_seconds"), "mean"
        ].max()
        max_rss = current.loc[current["metric"].eq("peak_rss_mb"), "mean"].max()
        print(
            f"n={int(n)}: exact-block mean ARI={exact_ari:.3f}; "
            f"minimum PPJoin recall={ppjoin_recall:.3f}; "
            f"sparsity={sparsity:.3f}; pair retention={pair_retention:.3f}; "
            f"null mean false positives/level={null_false_positives:.3f}; "
            f"pair-only recall={pair_only_recall:.3f}; "
            f"full pair recall={full_pair_recall:.3f}; "
            f"max runtime={max_runtime:.1f}s, max RSS={max_rss:.1f} MB"
        )
    print(f"\nSaved report to {root / 'REPORT.md'}")


def main():
    configuration, preset_name, preset = load_configuration(CONFIG_PATH, PRESET_NAME)
    settings = configuration["evaluation"]
    data_root = PROJECT / configuration["paths"]["data_root"] / preset_name
    output_root = PROJECT / configuration["paths"]["evaluation_root"] / preset_name
    jobs = []
    for n in map(int, preset["sample_sizes"]):
        for seed in map(int, preset["seeds"]):
            for scenario in configuration["scenarios"]:
                dataset = data_root / f"n_{n}" / f"seed_{seed}" / scenario
                if not (dataset / "tensor.npz").exists():
                    raise FileNotFoundError(
                        f"Generate the {preset_name} preset first; missing {dataset}"
                    )
                output = output_root / f"n_{n}" / f"seed_{seed}" / scenario
                jobs.append((str(dataset), str(output), settings))
    workers = synthetic_worker_count(len(jobs))
    print(f"Evaluating {len(jobs)} datasets with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(
        processes=workers, maxtasksperchild=1
    ) as pool:
        pool.starmap(evaluate_worker, jobs)
    aggregate_report(output_root, preset_name)


if __name__ == "__main__":
    main()
