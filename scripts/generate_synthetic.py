#!/usr/bin/env python3
"""Generate clustered sparse tensors first, then logistic binary targets."""

from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SYNTHETIC_CONFIG", PROJECT / "synthetic_config.json"))
PRESET_NAME = os.environ.get("SYNTHETIC_PRESET")

sys.path.insert(0, str(PROJECT))
from research_utils import (  # noqa: E402
    aggregate_columns,
    grouped_residual_features,
    marginal_matrix,
    normalize_rows,
    sqrt_sparse,
)


def load_configuration(path=CONFIG_PATH, preset_name=PRESET_NAME):
    with Path(path).open(encoding="utf-8") as handle:
        configuration = json.load(handle)
    name = preset_name or configuration["active_preset"]
    if name not in configuration["presets"]:
        raise ValueError(f"Unknown synthetic preset: {name}")
    preset = configuration["presets"][name]
    validate_preset(preset)
    return configuration, name, preset


def synthetic_worker_count(job_count):
    """Return the requested process count, capped by the available jobs."""
    try:
        workers = int(os.environ.get("SYNTHETIC_WORKERS", "5"))
    except ValueError as error:
        raise ValueError("SYNTHETIC_WORKERS must be a positive integer") from error
    if workers < 1:
        raise ValueError("SYNTHETIC_WORKERS must be a positive integer")
    return min(workers, job_count)


def validate_preset(preset):
    p = int(preset["bacteria"])
    q = int(preset["genes"])
    r = int(preset["bacteria_clusters"])
    c = int(preset["gene_clusters"])
    if min(p, q, r, c, *preset["sample_sizes"]) <= 0:
        raise ValueError("Synthetic dimensions must be positive")
    if p % r or q % c:
        raise ValueError("Bacteria and genes must divide evenly into clusters")

    levels = np.asarray(preset["block_prevalence_levels"], dtype=float)
    weights = np.asarray(preset["block_prevalence_weights"], dtype=float)
    if len(levels) != len(weights) or not len(levels):
        raise ValueError("Block prevalence levels and weights must have equal length")
    if np.any((levels <= 0) | (levels > 1)):
        raise ValueError("Block prevalence levels must be in (0, 1]")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Block prevalence weights must be nonnegative and sum to one")

    probabilities = [
        preset["member_dropout"],
        preset["target_prevalence"],
        preset["signal_pair_min_block_prevalence"],
    ]
    if any(not 0 < float(value) < 1 for value in probabilities):
        raise ValueError("Dropout, target, and signal prevalence must be in (0, 1)")
    pair_count = int(preset["pair_count"])
    if pair_count <= 0 or pair_count % 2 or pair_count > min(p, q):
        raise ValueError("Pair count must be positive, even, and no larger than either mode")
    for cluster in map(int, preset["bacteria_effects"]):
        if cluster < 0 or cluster >= r:
            raise ValueError(f"Unknown bacteria signal cluster: {cluster}")
    for cluster in map(int, preset["gene_effects"]):
        if cluster < 0 or cluster >= c:
            raise ValueError(f"Unknown gene signal cluster: {cluster}")


def balanced_memberships(size, clusters, rng):
    labels = np.repeat(np.arange(clusters, dtype=np.int32), size // clusters)
    rng.shuffle(labels)
    return labels


def random_streams(seed, n_samples):
    names = [
        "membership",
        "prevalence",
        "latent",
        "detection",
        "dropout",
        "noise",
        "pairs",
        "target",
    ]
    children = np.random.SeedSequence([int(seed), int(n_samples)]).spawn(len(names))
    seeds = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children)
    }
    return seeds, {name: np.random.default_rng(value) for name, value in seeds.items()}


def make_block_prevalence(preset, rng):
    levels = np.asarray(preset["block_prevalence_levels"], dtype=float)
    weights = np.asarray(preset["block_prevalence_weights"], dtype=float)
    shape = (int(preset["bacteria_clusters"]), int(preset["gene_clusters"]))
    expected_counts = weights * np.prod(shape)
    counts = np.floor(expected_counts).astype(int)
    remainder = int(np.prod(shape) - counts.sum())
    order = np.argsort(-(expected_counts - counts), kind="stable")
    counts[order[:remainder]] += 1
    values = np.repeat(levels, counts)
    rng.shuffle(values)
    return values.reshape(shape)


def make_latent(n, preset, block_prevalence, latent_rng, detection_rng):
    r, c = block_prevalence.shape
    detected = detection_rng.random((n, r, c)) < block_prevalence
    for sample in np.flatnonzero(~detected.any(axis=(1, 2))):
        while not detected[sample].any():
            detected[sample] = detection_rng.random((r, c)) < block_prevalence
    return {
        "detected": detected,
        "a": latent_rng.normal(0, float(preset["sigma_bacteria"]), size=(n, r)),
        "g": latent_rng.normal(0, float(preset["sigma_gene"]), size=(n, c)),
    }


def sorted_coordinates(sample, bacterium, gene, value):
    order = np.lexsort((gene, bacterium, sample))
    return {
        "sample": np.asarray(sample, dtype=np.int32)[order],
        "bacterium": np.asarray(bacterium, dtype=np.int32)[order],
        "gene": np.asarray(gene, dtype=np.int32)[order],
        "value": np.asarray(value, dtype=np.float32)[order],
    }


def generate_coordinates(
    latent,
    bacteria_membership,
    gene_membership,
    preset,
    dropout_rng,
    noise_rng,
    exact=False,
):
    bacteria_by_cluster = [
        np.flatnonzero(bacteria_membership == cluster)
        for cluster in range(int(preset["bacteria_clusters"]))
    ]
    genes_by_cluster = [
        np.flatnonzero(gene_membership == cluster)
        for cluster in range(int(preset["gene_clusters"]))
    ]
    sample_parts, bacteria_parts, gene_parts, value_parts = [], [], [], []
    dropout = 0.0 if exact else float(preset["member_dropout"])
    sigma_member = 0.0 if exact else float(preset["sigma_member"])
    mu = float(preset["mu"])

    for sample in range(latent["detected"].shape[0]):
        for bacteria_cluster, gene_cluster in np.argwhere(latent["detected"][sample]):
            members_b = bacteria_by_cluster[bacteria_cluster]
            members_g = genes_by_cluster[gene_cluster]
            bacteria = np.repeat(members_b, len(members_g))
            genes = np.tile(members_g, len(members_b))
            if dropout:
                keep = dropout_rng.random(len(bacteria)) >= dropout
                bacteria, genes = bacteria[keep], genes[keep]
            if not len(bacteria):
                continue
            noise = (
                noise_rng.normal(0, sigma_member, len(bacteria))
                if sigma_member
                else np.zeros(len(bacteria))
            )
            log_value = (
                mu
                + latent["a"][sample, bacteria_cluster]
                + latent["g"][sample, gene_cluster]
                + noise
            )
            sample_parts.append(np.full(len(bacteria), sample, dtype=np.int32))
            bacteria_parts.append(bacteria.astype(np.int32, copy=False))
            gene_parts.append(genes.astype(np.int32, copy=False))
            value_parts.append(np.exp(log_value).astype(np.float32, copy=False))

    coordinates = sorted_coordinates(
        np.concatenate(sample_parts),
        np.concatenate(bacteria_parts),
        np.concatenate(gene_parts),
        np.concatenate(value_parts),
    )
    if not np.isfinite(coordinates["value"]).all() or np.any(coordinates["value"] <= 0):
        raise ValueError("Generated values must be finite and positive")
    return coordinates


def coordinates_to_csr(coordinates, shape):
    columns = coordinates["bacterium"].astype(np.int64) * shape[2] + coordinates["gene"]
    return sparse.coo_matrix(
        (coordinates["value"], (coordinates["sample"], columns)),
        shape=(shape[0], shape[1] * shape[2]),
    ).tocsr()


def standardize(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("A selected target predictor has zero variance")
    return (values - mean) / scale, mean, scale


def choose_pairs(
    matrix,
    bacteria_membership,
    gene_membership,
    block_prevalence,
    preset,
    rng,
):
    p, q = int(preset["bacteria"]), int(preset["genes"])
    pair_bacteria = np.repeat(np.arange(p, dtype=np.int32), q)
    pair_genes = np.tile(np.arange(q, dtype=np.int32), p)
    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    threshold = max(5, int(np.ceil(0.05 * matrix.shape[0])))
    excluded_bacteria = np.array(list(map(int, preset["bacteria_effects"])))
    excluded_genes = np.array(list(map(int, preset["gene_effects"])))
    eligible = (
        ~np.isin(bacteria_membership[pair_bacteria], excluded_bacteria)
        & ~np.isin(gene_membership[pair_genes], excluded_genes)
        & (
            block_prevalence[
                bacteria_membership[pair_bacteria], gene_membership[pair_genes]
            ]
            >= float(preset["signal_pair_min_block_prevalence"])
        )
        & (support >= threshold)
    )
    candidates = np.flatnonzero(eligible)
    rng.shuffle(candidates)
    selected, used_bacteria, used_genes = [], set(), set()
    for pair in candidates:
        bacterium, gene = int(pair_bacteria[pair]), int(pair_genes[pair])
        if bacterium in used_bacteria or gene in used_genes:
            continue
        selected.append(int(pair))
        used_bacteria.add(bacterium)
        used_genes.add(gene)
        if len(selected) == int(preset["pair_count"]):
            break
    if len(selected) != int(preset["pair_count"]):
        raise ValueError("Not enough prevalent distinct pairs outside marginal clusters")
    effects = np.tile(
        [-float(preset["pair_effect"]), float(preset["pair_effect"])],
        len(selected) // 2,
    )
    rng.shuffle(effects)
    return np.asarray(selected, dtype=np.int64), effects


def logistic_target(score, target_prevalence, uniforms):
    score = np.asarray(score, dtype=float)
    low, high = -50.0, 50.0
    for _ in range(80):
        middle = (low + high) / 2
        if expit(score + middle).mean() < target_prevalence:
            low = middle
        else:
            high = middle
    intercept = (low + high) / 2
    linear_predictor = score + intercept
    probability = expit(linear_predictor)
    target = (uniforms < probability).astype(np.int8)
    if len(np.unique(target)) != 2:
        raise ValueError("Generated logistic target contains only one class")
    return target, linear_predictor, probability, intercept


def build_targets(
    matrix,
    bacteria_membership,
    gene_membership,
    block_prevalence,
    preset,
    pair_rng,
    target_rng,
):
    n, p, q = matrix.shape[0], int(preset["bacteria"]), int(preset["genes"])
    normalized = normalize_rows(matrix)
    pair_bacteria = np.repeat(np.arange(p, dtype=np.int32), q)
    pair_genes = np.tile(np.arange(q, dtype=np.int32), p)
    bacteria = marginal_matrix(normalized, pair_bacteria, p)
    genes = marginal_matrix(normalized, pair_genes, q)
    bacteria_clusters, _ = aggregate_columns(bacteria, bacteria_membership)
    gene_clusters, _ = aggregate_columns(genes, gene_membership)

    bacteria_ids = np.asarray(sorted(map(int, preset["bacteria_effects"])), dtype=int)
    gene_ids = np.asarray(sorted(map(int, preset["gene_effects"])), dtype=int)
    bacteria_beta = np.asarray(
        [float(preset["bacteria_effects"][str(cluster)]) for cluster in bacteria_ids]
    )
    gene_beta = np.asarray(
        [float(preset["gene_effects"][str(cluster)]) for cluster in gene_ids]
    )
    bacteria_z, bacteria_mean, bacteria_sd = standardize(
        sqrt_sparse(bacteria_clusters[:, bacteria_ids]).toarray()
    )
    gene_z, gene_mean, gene_sd = standardize(
        sqrt_sparse(gene_clusters[:, gene_ids]).toarray()
    )
    selected_pairs, pair_beta = choose_pairs(
        matrix,
        bacteria_membership,
        gene_membership,
        block_prevalence,
        preset,
        pair_rng,
    )
    pair_groups = (
        bacteria_membership[pair_bacteria] * int(preset["gene_clusters"])
        + gene_membership[pair_genes]
    )
    transformed_pairs = sqrt_sparse(normalized)
    pair_values, _ = grouped_residual_features(
        transformed_pairs,
        transformed_pairs,
        pair_groups,
        selected_pairs,
    )
    pair_values = pair_values.toarray()
    pair_z, pair_mean, pair_sd = standardize(pair_values)

    marginal_score = bacteria_z @ bacteria_beta + gene_z @ gene_beta
    pair_score = pair_z @ pair_beta
    full_score = marginal_score + pair_score
    uniforms = target_rng.random(n)
    target_prevalence = float(preset["target_prevalence"])
    targets = {
        "null_target": logistic_target(np.zeros(n), target_prevalence, uniforms),
        "marginal_signals": logistic_target(marginal_score, target_prevalence, uniforms),
        "pair_signals": logistic_target(pair_score, target_prevalence, uniforms),
        "full_signals": logistic_target(full_score, target_prevalence, uniforms),
    }

    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    marginal_rows = []
    for level, ids, effects, means, scales, values in (
        ("bacterium", bacteria_ids, bacteria_beta, bacteria_mean, bacteria_sd, bacteria_clusters),
        ("gene", gene_ids, gene_beta, gene_mean, gene_sd, gene_clusters),
    ):
        transformed = sqrt_sparse(values[:, ids]).toarray()
        for position, (cluster, effect) in enumerate(zip(ids, effects)):
            marginal_rows.append({
                "level": level,
                "cluster": int(cluster),
                "bacterium": pd.NA,
                "gene": pd.NA,
                "effect": float(effect),
                "direction": "positive" if effect > 0 else "negative",
                "standardization_mean": float(means[position]),
                "standardization_sd": float(scales[position]),
                "expected_prevalence": pd.NA,
                "realized_support": int(np.count_nonzero(transformed[:, position])),
            })
    pair_rows = []
    for position, (pair, effect) in enumerate(zip(selected_pairs, pair_beta)):
        bacterium, gene = divmod(int(pair), q)
        expected = block_prevalence[
            bacteria_membership[bacterium], gene_membership[gene]
        ] * (1 - float(preset["member_dropout"]))
        pair_rows.append({
            "level": "pair",
            "cluster": pd.NA,
            "bacterium": bacterium,
            "gene": gene,
            "effect": float(effect),
            "direction": "positive" if effect > 0 else "negative",
            "standardization_mean": float(pair_mean[position]),
            "standardization_sd": float(pair_sd[position]),
            "expected_prevalence": float(expected),
            "realized_support": int(support[pair]),
        })
    return targets, marginal_rows, pair_rows


def block_truth_rows(block_prevalence, latent, preset):
    rows = []
    for bacteria_cluster, gene_cluster in np.ndindex(block_prevalence.shape):
        prevalence = float(block_prevalence[bacteria_cluster, gene_cluster])
        rows.append({
            "bacteria_cluster": bacteria_cluster,
            "gene_cluster": gene_cluster,
            "expected_prevalence": prevalence,
            "noisy_expected_prevalence": prevalence
            * (1 - float(preset["member_dropout"])),
            "realized_block_support": int(
                latent["detected"][:, bacteria_cluster, gene_cluster].sum()
            ),
        })
    return rows


def save_dataset(
    output,
    coordinates,
    tensor_source,
    shape,
    target,
    target_model,
    bacteria_membership,
    gene_membership,
    block_rows,
    signal_rows,
    metadata,
):
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "tensor.npz"
    tensor_path.unlink(missing_ok=True)
    if tensor_source is not None:
        os.link(tensor_source, tensor_path)
    else:
        index_types = [
            np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32
            for maximum in shape
        ]
        np.savez_compressed(
            tensor_path,
            sample=coordinates["sample"].astype(index_types[0]),
            bacterium=coordinates["bacterium"].astype(index_types[1]),
            gene=coordinates["gene"].astype(index_types[2]),
            value=coordinates["value"].astype(np.float32),
            shape=np.asarray(shape, dtype=np.int64),
        )

    target_path = output / "target.npy"
    model_path = output / "target_model.npz"
    if target is None:
        target_path.unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)
    else:
        np.save(target_path, np.asarray(target, dtype=np.int8))
        np.savez_compressed(
            model_path,
            linear_predictor=np.asarray(target_model[0], dtype=np.float64),
            probability=np.asarray(target_model[1], dtype=np.float64),
        )
    pd.DataFrame({
        "bacterium": np.arange(shape[1]),
        "true_cluster": bacteria_membership,
    }).to_csv(output / "bacteria_truth.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "gene": np.arange(shape[2]),
        "true_cluster": gene_membership,
    }).to_csv(output / "gene_truth.csv.gz", index=False, compression="gzip")
    pd.DataFrame(block_rows).to_csv(
        output / "block_truth.csv.gz", index=False, compression="gzip"
    )
    signal_columns = [
        "level",
        "cluster",
        "bacterium",
        "gene",
        "effect",
        "direction",
        "standardization_mean",
        "standardization_sd",
        "expected_prevalence",
        "realized_support",
    ]
    pd.DataFrame(signal_rows, columns=signal_columns).to_csv(
        output / "signal_truth.csv.gz", index=False, compression="gzip"
    )
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_tensor(path):
    with np.load(Path(path) / "tensor.npz") as stored:
        shape = tuple(map(int, stored["shape"]))
        return (
            stored["sample"].astype(np.int64),
            stored["bacterium"].astype(np.int64),
            stored["gene"].astype(np.int64),
            stored["value"].astype(np.float64),
            shape,
        )


def tensor_to_csr(path):
    sample, bacterium, gene, value, shape = load_tensor(path)
    columns = bacterium * shape[2] + gene
    matrix = sparse.coo_matrix(
        (value, (sample, columns)), shape=(shape[0], shape[1] * shape[2])
    ).tocsr()
    edge_bacteria = np.repeat(np.arange(shape[1], dtype=np.int32), shape[2])
    edge_gene = np.tile(np.arange(shape[2], dtype=np.int32), shape[1])
    return matrix, edge_bacteria, edge_gene, shape


def scenario_metadata(
    configuration,
    preset_name,
    preset,
    scenario,
    seed,
    stream_seeds,
    shape,
    nnz,
    block_prevalence,
    config_hash,
    target_result=None,
):
    exact = scenario == "exact_blocks"
    expected_density = float(block_prevalence.mean())
    if not exact:
        expected_density *= 1 - float(preset["member_dropout"])
    metadata = {
        "format_version": configuration["format_version"],
        "preset": preset_name,
        "scenario": scenario,
        "seed": int(seed),
        "stream_seeds": stream_seeds,
        "shape": list(shape),
        "nnz": int(nnz),
        "realized_density": float(nnz / np.prod(shape)),
        "realized_sparsity": float(1 - nnz / np.prod(shape)),
        "expected_density": expected_density,
        "expected_sparsity": 1 - expected_density,
        "expected_within_block_jaccard": 1.0 if exact else float(
            (1 - preset["member_dropout"]) / (1 + preset["member_dropout"])
        ),
        "block_prevalence_min": float(block_prevalence.min()),
        "block_prevalence_max": float(block_prevalence.max()),
        "has_target": target_result is not None,
        "configuration_sha256": config_hash,
        "configuration": configuration,
        "resolved_preset": preset,
    }
    if target_result is not None:
        target, _, probability, intercept = target_result
        metadata.update({
            "target_intercept": float(intercept),
            "expected_positive_fraction": float(probability.mean()),
            "realized_positive_fraction": float(target.mean()),
        })
    return metadata


def generate_size(configuration, preset_name, preset, n, seed):
    started = perf_counter()
    stream_seeds, rng = random_streams(seed, n)
    bacteria_membership = balanced_memberships(
        int(preset["bacteria"]), int(preset["bacteria_clusters"]), rng["membership"]
    )
    gene_membership = balanced_memberships(
        int(preset["genes"]), int(preset["gene_clusters"]), rng["membership"]
    )
    block_prevalence = make_block_prevalence(preset, rng["prevalence"])
    latent = make_latent(
        n, preset, block_prevalence, rng["latent"], rng["detection"]
    )
    shape = (n, int(preset["bacteria"]), int(preset["genes"]))
    exact = generate_coordinates(
        latent,
        bacteria_membership,
        gene_membership,
        preset,
        rng["dropout"],
        rng["noise"],
        exact=True,
    )
    noisy = generate_coordinates(
        latent,
        bacteria_membership,
        gene_membership,
        preset,
        rng["dropout"],
        rng["noise"],
    )
    noisy_matrix = coordinates_to_csr(noisy, shape)
    targets, marginal_rows, pair_rows = build_targets(
        noisy_matrix,
        bacteria_membership,
        gene_membership,
        block_prevalence,
        preset,
        rng["pairs"],
        rng["target"],
    )
    block_rows = block_truth_rows(block_prevalence, latent, preset)
    run_root = (
        PROJECT
        / configuration["paths"]["data_root"]
        / preset_name
        / f"n_{n}"
        / f"seed_{seed}"
    )
    config_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    for scenario, coordinates in (("exact_blocks", exact), ("noisy_blocks", noisy)):
        metadata = scenario_metadata(
            configuration,
            preset_name,
            preset,
            scenario,
            seed,
            stream_seeds,
            shape,
            len(coordinates["value"]),
            block_prevalence,
            config_hash,
        )
        save_dataset(
            run_root / scenario,
            coordinates,
            None,
            shape,
            None,
            None,
            bacteria_membership,
            gene_membership,
            block_rows,
            [],
            metadata,
        )

    noisy_tensor = run_root / "noisy_blocks" / "tensor.npz"
    for scenario in (
        "null_target", "marginal_signals", "pair_signals", "full_signals"
    ):
        target_result = targets[scenario]
        signal_rows = (
            marginal_rows if scenario in {"marginal_signals", "full_signals"} else []
        )
        if scenario in {"pair_signals", "full_signals"}:
            signal_rows = signal_rows + pair_rows
        metadata = scenario_metadata(
            configuration,
            preset_name,
            preset,
            scenario,
            seed,
            stream_seeds,
            shape,
            len(noisy["value"]),
            block_prevalence,
            config_hash,
            target_result,
        )
        target, linear_predictor, probability, _ = target_result
        save_dataset(
            run_root / scenario,
            None,
            noisy_tensor,
            shape,
            target,
            (linear_predictor, probability),
            bacteria_membership,
            gene_membership,
            block_rows,
            signal_rows,
            metadata,
        )

    for scenario in configuration["scenarios"]:
        metadata = json.loads((run_root / scenario / "metadata.json").read_text())
        print(
            f"{preset_name} n={n} seed={seed} {scenario}: "
            f"nnz={metadata['nnz']:,}; sparsity={metadata['realized_sparsity']:.3f}",
            flush=True,
        )
    print(f"Generated n={n}, seed={seed} in {perf_counter() - started:.1f}s", flush=True)


def main():
    configuration, preset_name, preset = load_configuration()
    jobs = [
        (configuration, preset_name, preset, n, seed)
        for n in map(int, preset["sample_sizes"])
        for seed in map(int, preset["seeds"])
    ]
    workers = synthetic_worker_count(len(jobs))
    print(f"Generating {len(jobs)} configurations with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(
        processes=workers, maxtasksperchild=1
    ) as pool:
        pool.starmap(generate_size, jobs)


if __name__ == "__main__":
    main()
