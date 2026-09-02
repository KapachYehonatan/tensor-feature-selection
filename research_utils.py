"""Small shared calculations for the local tensor feature-selection scripts."""

from __future__ import annotations

import gzip
from array import array
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse, stats


def benjamini_hochberg(p_values):
    """Return BH-adjusted q-values in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.ones_like(p_values)
    finite = np.isfinite(p_values)
    if not finite.any():
        return q_values
    p = p_values[finite]
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(ranked)
    restored[order] = ranked
    q_values[finite] = restored
    return q_values


def read_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_lines(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def sample_support(matrix):
    """Count the samples with a positive value for each feature."""
    matrix = sparse.csr_matrix(matrix)
    detected = matrix.copy()
    detected.data = detected.data > 0
    detected.eliminate_zeros()
    return np.asarray(detected.getnnz(axis=0)).ravel().astype(np.int32)


def prevalence_mask(matrix, fraction=0.05, minimum_samples=5):
    if not 0 < fraction <= 1:
        raise ValueError("Prevalence fraction must be in (0, 1]")
    threshold = max(minimum_samples, int(np.ceil(fraction * matrix.shape[0])))
    support = sample_support(matrix)
    return support >= threshold, support, threshold


def normalize_rows(matrix):
    """Close each nonempty sample to unit retained mass."""
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError("Abundances must be nonnegative")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError(f"{np.count_nonzero(totals <= 0)} samples have no retained abundance")
    return sparse.diags(1 / totals) @ matrix


def sqrt_sparse(matrix):
    transformed = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    transformed.data = np.sqrt(transformed.data)
    return transformed


def prediction_matrix(values, n_rows):
    """Keep a single predicted target as an n-by-1 matrix."""
    return np.asarray(values).reshape(n_rows, -1)


def marginal_matrix(matrix, pair_entity, n_entities):
    """Sum pair columns into bacteria or gene marginal columns."""
    pair_entity = np.asarray(pair_entity, dtype=np.int64)
    if matrix.shape[1] != len(pair_entity):
        raise ValueError("Pair mapping does not match matrix columns")
    mapping = sparse.csr_matrix(
        (np.ones(len(pair_entity)), (np.arange(len(pair_entity)), pair_entity)),
        shape=(len(pair_entity), n_entities),
    )
    return (matrix @ mapping).tocsr()


def contextual_profiles(matrix, pair_entity, pair_context, n_entities, n_contexts):
    """Return entity columns over the preserved sample x context coordinates."""
    coo = sparse.coo_matrix(matrix)
    pair_entity = np.asarray(pair_entity, dtype=np.int64)
    pair_context = np.asarray(pair_context, dtype=np.int64)
    rows = coo.row.astype(np.int64) * n_contexts + pair_context[coo.col]
    columns = pair_entity[coo.col]
    return sparse.csr_matrix(
        (coo.data, (rows, columns)),
        shape=(matrix.shape[0] * n_contexts, n_entities),
    )


def aggregate_columns(matrix, labels):
    """Sum original columns belonging to each similarity cluster."""
    labels = np.asarray(labels, dtype=np.int64)
    unique, compact = np.unique(labels, return_inverse=True)
    mapping = sparse.csr_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), compact)),
        shape=(len(labels), len(unique)),
    )
    return (matrix @ mapping).tocsr(), compact


def _support_overlap(csc, left, right):
    """Count shared nonzero coordinates with a small vectorized lookup."""
    start_l, end_l = csc.indptr[left : left + 2]
    start_r, end_r = csc.indptr[right : right + 2]
    index_l, index_r = csc.indices[start_l:end_l], csc.indices[start_r:end_r]
    if len(index_l) > len(index_r):
        index_l, index_r = index_r, index_l
    positions = np.searchsorted(index_r, index_l)
    valid = positions < len(index_r)
    return int(np.count_nonzero(index_r[positions[valid]] == index_l[valid]))


def _column_similarity(csc, left, right, metric, column_sums=None, column_squares=None):
    """Calculate exact sparse abundance similarity and detection Jaccard."""
    start_l, end_l = csc.indptr[left : left + 2]
    start_r, end_r = csc.indptr[right : right + 2]
    index_l, value_l = csc.indices[start_l:end_l], csc.data[start_l:end_l]
    index_r, value_r = csc.indices[start_r:end_r], csc.data[start_r:end_r]
    pos_l = pos_r = common = 0
    cross = shared_min = 0.0
    while pos_l < len(index_l) and pos_r < len(index_r):
        if index_l[pos_l] == index_r[pos_r]:
            common += 1
            cross += value_l[pos_l] * value_r[pos_r]
            shared_min += min(value_l[pos_l], value_r[pos_r])
            pos_l += 1
            pos_r += 1
        elif index_l[pos_l] < index_r[pos_r]:
            pos_l += 1
        else:
            pos_r += 1
    union = len(index_l) + len(index_r) - common
    jaccard = common / union if union else 1.0

    sum_l = value_l.sum() if column_sums is None else column_sums[left]
    sum_r = value_r.sum() if column_sums is None else column_sums[right]

    if metric == "bray_curtis":
        denominator = sum_l + sum_r
        similarity = 2 * shared_min / denominator if denominator else 1.0
        return similarity, jaccard
    if metric != "correlation":
        raise ValueError("Metric must be 'correlation' or 'bray_curtis'")

    n = csc.shape[0]
    square_l = np.dot(value_l, value_l) if column_squares is None else column_squares[left]
    square_r = np.dot(value_r, value_r) if column_squares is None else column_squares[right]
    var_l = square_l - sum_l * sum_l / n
    var_r = square_r - sum_r * sum_r / n
    if var_l <= 1e-15 or var_r <= 1e-15:
        identical = np.array_equal(index_l, index_r) and np.allclose(value_l, value_r)
        return float(identical), jaccard
    correlation = (cross - sum_l * sum_r / n) / np.sqrt(var_l * var_r)
    return float(np.clip(correlation, -1, 1)), jaccard


def similarity_clusters(
    profiles,
    similarity_threshold=0.95,
    jaccard_threshold=0.80,
    metric="correlation",
    return_diagnostics=False,
    collect_checked_pairs=False,
    verbose=False,
):
    """Assign columns to directly similar representatives, without choosing K."""
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")
    if not 0 < jaccard_threshold <= 1:
        raise ValueError("Jaccard threshold must be in (0, 1]")
    if metric not in {"correlation", "bray_curtis"}:
        raise ValueError("Metric must be 'correlation' or 'bray_curtis'")
    csc = sparse.csc_matrix(profiles, dtype=np.float64)
    csc.sum_duplicates()
    csc.eliminate_zeros()
    csc.sort_indices()
    if csc.data.size and (not np.isfinite(csc.data).all() or csc.data.min() < 0):
        raise ValueError("Similarity profiles must contain finite nonnegative values")

    started = perf_counter()
    n_features = csc.shape[1]
    support = np.diff(csc.indptr).astype(np.int64, copy=False)
    empty = np.flatnonzero(support == 0)
    if len(empty):
        raise ValueError(f"{len(empty)} similarity features have zero support")

    diagnostics = {
        "features": n_features,
        "coordinates": csc.shape[0],
        "profile_nnz": csc.nnz,
        "indexed_prefix_entries": 0,
        "posting_visits": 0,
        "size_rejections": 0,
        "positional_rejections": 0,
        "exact_jaccard_checks": 0,
        "jaccard_passes": 0,
        "abundance_similarity_checks": 0,
        "accepted_assignments": 0,
        "representatives": 0,
    }
    checked_pairs = [] if collect_checked_pairs else None
    if not n_features:
        diagnostics["elapsed_seconds"] = perf_counter() - started
        result = np.empty(0, dtype=np.int64)
        return (result, diagnostics) if return_diagnostics else result

    # Cache column moments and rank coordinates from rarest to most prevalent.
    column_sums = np.add.reduceat(csc.data, csc.indptr[:-1])
    squared_data = np.square(csc.data)
    column_squares = np.add.reduceat(squared_data, csc.indptr[:-1])
    del squared_data
    frequency = np.bincount(csc.indices, minlength=csc.shape[0])
    coordinate_order = np.lexsort((np.arange(csc.shape[0]), frequency))
    coordinate_rank = np.empty(csc.shape[0], dtype=np.int64)
    coordinate_rank[coordinate_order] = np.arange(csc.shape[0])
    feature_order = np.lexsort((np.arange(n_features), support))

    # Each token indexes only current representatives: ids and prefix positions.
    postings = {}
    labels = np.full(n_features, -1, dtype=np.int64)
    next_label = 0
    indexed_prefix_entries = posting_visits = size_rejections = 0
    positional_rejections = exact_jaccard_checks = jaccard_passes = 0
    abundance_similarity_checks = accepted_assignments = 0
    for processed, feature in enumerate(feature_order, start=1):
        start, end = csc.indptr[feature : feature + 2]
        ordered = csc.indices[start:end]
        ordered = ordered[np.argsort(coordinate_rank[ordered])]
        size = len(ordered)
        prefix_length = size - int(np.ceil(jaccard_threshold * size)) + 1
        prefix = ordered[:prefix_length]
        minimum_size = int(np.ceil(jaccard_threshold * size))

        overlap_counts = {}
        for position, coordinate in enumerate(prefix):
            posting = postings.get(int(coordinate))
            if posting is None:
                continue
            representative_ids, representative_positions = posting
            # Representatives were indexed in support-size order, so skip the
            # too-short prefix of this posting list with a small binary search.
            low, high = 0, len(representative_ids)
            while low < high:
                middle = (low + high) // 2
                if support[representative_ids[middle]] < minimum_size:
                    low = middle + 1
                else:
                    high = middle
            size_rejections += low
            for posting_index in range(low, len(representative_ids)):
                representative = int(representative_ids[posting_index])
                posting_visits += 1
                count = overlap_counts.get(representative, 0)
                if count < 0:
                    continue
                representative_size = support[representative]
                required = int(np.ceil(
                    jaccard_threshold / (1 + jaccard_threshold)
                    * (size + representative_size)
                    - 1e-12
                ))
                remaining = min(
                    size - position - 1,
                    representative_size - int(representative_positions[posting_index]) - 1,
                )
                if count + 1 + remaining >= required:
                    overlap_counts[representative] = count + 1
                else:
                    overlap_counts[representative] = -1
                    positional_rejections += 1

        best_representative = None
        best_key = None
        for representative, prefix_overlap in overlap_counts.items():
            if prefix_overlap < 0:
                continue
            exact_jaccard_checks += 1
            if checked_pairs is not None:
                checked_pairs.append((int(feature), int(representative)))
            common = _support_overlap(csc, feature, representative)
            union = size + support[representative] - common
            jaccard = common / union
            if jaccard < jaccard_threshold:
                continue
            jaccard_passes += 1
            abundance_similarity_checks += 1
            similarity, _ = _column_similarity(
                csc,
                feature,
                representative,
                metric,
                column_sums,
                column_squares,
            )
            if similarity < similarity_threshold:
                continue
            key = (similarity, jaccard, -representative)
            if best_key is None or key > best_key:
                best_key = key
                best_representative = representative

        if best_representative is None:
            labels[feature] = next_label
            next_label += 1
            for position, coordinate in enumerate(prefix):
                representative_ids, representative_positions = postings.setdefault(
                    int(coordinate), (array("I"), array("I"))
                )
                representative_ids.append(feature)
                representative_positions.append(position)
            indexed_prefix_entries += prefix_length
        else:
            labels[feature] = labels[best_representative]
            accepted_assignments += 1

        if verbose and processed % 10_000 == 0:
            print(
                f"Similarity clustering: {processed:,}/{n_features:,} features; "
                f"{next_label:,} representatives; "
                f"{abundance_similarity_checks:,} exact similarities",
                flush=True,
            )

    diagnostics.update({
        "indexed_prefix_entries": indexed_prefix_entries,
        "posting_visits": posting_visits,
        "size_rejections": size_rejections,
        "positional_rejections": positional_rejections,
        "exact_jaccard_checks": exact_jaccard_checks,
        "jaccard_passes": jaccard_passes,
        "abundance_similarity_checks": abundance_similarity_checks,
        "accepted_assignments": accepted_assignments,
        "representatives": next_label,
        "elapsed_seconds": perf_counter() - started,
    })
    if checked_pairs is not None:
        diagnostics["checked_pairs"] = checked_pairs
    if verbose:
        print(
            f"Similarity clustering complete: {n_features:,} features -> "
            f"{next_label:,} representatives; {diagnostics['posting_visits']:,} "
            f"posting visits; {diagnostics['exact_jaccard_checks']:,} exact Jaccard; "
            f"{diagnostics['abundance_similarity_checks']:,} exact similarities; "
            f"{diagnostics['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return (labels, diagnostics) if return_diagnostics else labels


def exhaustive_similarity_clusters(
    profiles,
    similarity_threshold=0.95,
    jaccard_threshold=0.80,
    metric="correlation",
):
    """Small exhaustive reference for the representative clustering rule."""
    csc = sparse.csc_matrix(profiles, dtype=np.float64)
    csc.sum_duplicates()
    csc.eliminate_zeros()
    csc.sort_indices()
    support = np.diff(csc.indptr)
    if np.any(support == 0):
        raise ValueError("Exhaustive comparison received a zero-support feature")

    labels = np.full(csc.shape[1], -1, dtype=np.int64)
    representatives = []
    qualifying_pairs = []
    feature_order = np.lexsort((np.arange(csc.shape[1]), support))
    for feature in feature_order:
        best_representative = None
        best_key = None
        for representative in representatives:
            similarity, jaccard = _column_similarity(
                csc, feature, representative, metric
            )
            if jaccard >= jaccard_threshold:
                qualifying_pairs.append((int(feature), int(representative)))
            if jaccard < jaccard_threshold or similarity < similarity_threshold:
                continue
            key = (similarity, jaccard, -representative)
            if best_key is None or key > best_key:
                best_key = key
                best_representative = representative
        if best_representative is None:
            labels[feature] = len(representatives)
            representatives.append(int(feature))
        else:
            labels[feature] = labels[best_representative]
    return labels, qualifying_pairs


def similarity_coclustering(
    matrix,
    pair_bacteria,
    pair_gene,
    n_bacteria,
    n_genes,
    similarity_threshold=0.95,
    jaccard_threshold=0.80,
    metric="correlation",
    rounds=3,
    return_diagnostics=False,
    collect_checked_pairs=False,
    verbose=False,
):
    """Alternate conservative similarity clustering across the two modes."""
    active_bacteria, local_bacteria = np.unique(pair_bacteria, return_inverse=True)
    active_genes, local_genes = np.unique(pair_gene, return_inverse=True)
    if active_bacteria.max(initial=-1) >= n_bacteria or active_genes.max(initial=-1) >= n_genes:
        raise ValueError("Pair mapping contains an unknown entity index")

    bacteria_marginal = marginal_matrix(matrix, local_bacteria, len(active_bacteria))
    if verbose:
        print("Clustering bacteria marginals", flush=True)
    diagnostics = []
    bacteria_result = similarity_clusters(
        sqrt_sparse(bacteria_marginal), similarity_threshold,
        jaccard_threshold, metric, return_diagnostics=return_diagnostics,
        collect_checked_pairs=collect_checked_pairs, verbose=verbose,
    )
    if return_diagnostics:
        bacteria_labels, stage = bacteria_result
        stage["stage"] = "initial_bacteria"
        diagnostics.append(stage)
    else:
        bacteria_labels = bacteria_result
    gene_labels = np.arange(len(active_genes), dtype=np.int64)

    for round_index in range(rounds):
        if verbose:
            print(f"Co-clustering round {round_index + 1}: genes", flush=True)
        gene_profiles = contextual_profiles(
            matrix,
            local_genes,
            bacteria_labels[local_bacteria],
            len(active_genes),
            int(bacteria_labels.max()) + 1,
        )
        gene_result = similarity_clusters(
            sqrt_sparse(gene_profiles), similarity_threshold,
            jaccard_threshold, metric, return_diagnostics=return_diagnostics,
            collect_checked_pairs=collect_checked_pairs, verbose=verbose,
        )
        if return_diagnostics:
            new_gene_labels, stage = gene_result
            stage["stage"] = f"round_{round_index + 1}_genes"
            diagnostics.append(stage)
        else:
            new_gene_labels = gene_result
        if verbose:
            print(f"Co-clustering round {round_index + 1}: bacteria", flush=True)
        bacteria_profiles = contextual_profiles(
            matrix,
            local_bacteria,
            new_gene_labels[local_genes],
            len(active_bacteria),
            int(new_gene_labels.max()) + 1,
        )
        bacteria_result = similarity_clusters(
            sqrt_sparse(bacteria_profiles), similarity_threshold,
            jaccard_threshold, metric, return_diagnostics=return_diagnostics,
            collect_checked_pairs=collect_checked_pairs, verbose=verbose,
        )
        if return_diagnostics:
            new_bacteria_labels, stage = bacteria_result
            stage["stage"] = f"round_{round_index + 1}_bacteria"
            diagnostics.append(stage)
        else:
            new_bacteria_labels = bacteria_result
        stable = (
            np.array_equal(new_gene_labels, gene_labels)
            and np.array_equal(new_bacteria_labels, bacteria_labels)
        )
        gene_labels, bacteria_labels = new_gene_labels, new_bacteria_labels
        if stable:
            break

    result = active_bacteria, bacteria_labels, active_genes, gene_labels
    return (*result, diagnostics) if return_diagnostics else result


def mann_whitney_scores(matrix, y, chunk_size=2000):
    """Screen sparse columns in bounded dense chunks."""
    matrix = sparse.csc_matrix(matrix)
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Mann-Whitney screening requires binary labels 0 and 1")
    p_values = np.ones(matrix.shape[1])
    effects = np.zeros(matrix.shape[1])
    auc = np.full(matrix.shape[1], 0.5)
    positive, negative = y == 1, y == 0
    denominator = positive.sum() * negative.sum()
    for start in range(0, matrix.shape[1], chunk_size):
        end = min(start + chunk_size, matrix.shape[1])
        values = matrix[:, start:end].toarray()
        result = stats.mannwhitneyu(
            values[positive], values[negative], axis=0,
            alternative="two-sided", method="asymptotic",
        )
        statistic = np.asarray(result.statistic, dtype=float)
        p_values[start:end] = np.asarray(result.pvalue, dtype=float)
        auc[start:end] = statistic / denominator
        effects[start:end] = 2 * auc[start:end] - 1
    return p_values, benjamini_hochberg(p_values), effects, auc


def residual_correlation_scores(matrix, residual):
    """Calculate vectorized Pearson scores against a continuous residual."""
    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    residual = np.asarray(residual, dtype=float)
    n = len(residual)
    sum_x = np.asarray(matrix.sum(axis=0)).ravel()
    sum_x2 = np.asarray(matrix.power(2).sum(axis=0)).ravel()
    sum_y, sum_y2 = residual.sum(), np.dot(residual, residual)
    cross = np.asarray(matrix.T @ residual).ravel()
    numerator = cross - sum_x * sum_y / n
    denominator = np.sqrt(
        np.maximum(sum_x2 - sum_x * sum_x / n, 0)
        * max(sum_y2 - sum_y * sum_y / n, 0)
    )
    correlation = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    ).clip(-0.999999, 0.999999)
    statistic = correlation * np.sqrt((n - 2) / (1 - correlation * correlation))
    p_values = 2 * stats.t.sf(np.abs(statistic), df=max(n - 2, 1))
    return correlation, p_values, benjamini_hochberg(p_values)


def grouped_partial_correlation_scores(matrix, residual, groups):
    """Score each sparse feature after removing its group's aggregate signal."""
    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    residual = np.asarray(residual, dtype=float)
    groups = np.asarray(groups)
    if matrix.shape[0] != len(residual) or matrix.shape[1] != len(groups):
        raise ValueError("Matrix, residual, and group labels do not align")
    _, groups = np.unique(groups, return_inverse=True)
    totals = marginal_matrix(matrix, groups, int(groups.max()) + 1).tocsc()
    n = matrix.shape[0]

    sum_x = np.asarray(matrix.sum(axis=0)).ravel()
    sum_x2 = np.asarray(matrix.power(2).sum(axis=0)).ravel()
    sum_y, sum_y2 = residual.sum(), np.dot(residual, residual)
    cross_xy = np.asarray(matrix.T @ residual).ravel()
    sum_z = np.asarray(totals.sum(axis=0)).ravel()
    sum_z2 = np.asarray(totals.power(2).sum(axis=0)).ravel()
    cross_zy = np.asarray(totals.T @ residual).ravel()
    covariance_zy = cross_zy - sum_z * sum_y / n
    variance_z = sum_z2 - sum_z * sum_z / n

    covariance_xz = np.empty(matrix.shape[1])
    for group in range(totals.shape[1]):
        members = np.flatnonzero(groups == group)
        cross = (matrix[:, members].T @ totals[:, group]).toarray().ravel()
        covariance_xz[members] = cross - sum_x[members] * sum_z[group] / n

    variance_x = sum_x2 - sum_x * sum_x / n
    variance_y = sum_y2 - sum_y * sum_y / n
    covariance_xy = cross_xy - sum_x * sum_y / n
    valid_group = variance_z[groups] > 0
    partial_covariance = covariance_xy.copy()
    partial_variance_x = variance_x.copy()
    partial_variance_y = np.full(matrix.shape[1], variance_y)
    partial_covariance[valid_group] -= (
        covariance_xz[valid_group]
        * covariance_zy[groups[valid_group]]
        / variance_z[groups[valid_group]]
    )
    partial_variance_x[valid_group] -= (
        covariance_xz[valid_group] ** 2 / variance_z[groups[valid_group]]
    )
    partial_variance_y[valid_group] -= (
        covariance_zy[groups[valid_group]] ** 2 / variance_z[groups[valid_group]]
    )
    denominator = np.sqrt(
        np.maximum(partial_variance_x, 0) * np.maximum(partial_variance_y, 0)
    )
    correlation = np.divide(
        partial_covariance,
        denominator,
        out=np.zeros_like(partial_covariance),
        where=denominator > 0,
    ).clip(-0.999999, 0.999999)
    statistic = correlation * np.sqrt((n - 3) / (1 - correlation * correlation))
    p_values = 2 * stats.t.sf(np.abs(statistic), df=max(n - 3, 1))
    return correlation, p_values, benjamini_hochberg(p_values)


def grouped_residual_features(train, test, groups, columns):
    """Residualize selected columns against their group totals using train data."""
    train = sparse.csc_matrix(train, dtype=np.float64)
    test = sparse.csc_matrix(test, dtype=np.float64)
    groups = np.asarray(groups)
    columns = np.asarray(columns, dtype=int)
    if train.shape[1] != test.shape[1] or train.shape[1] != len(groups):
        raise ValueError("Train, test, and group labels do not align")
    if np.any((columns < 0) | (columns >= train.shape[1])):
        raise ValueError("Residual feature column is out of range")
    if not len(columns):
        return (
            sparse.csr_matrix((train.shape[0], 0)),
            sparse.csr_matrix((test.shape[0], 0)),
        )

    _, groups = np.unique(groups, return_inverse=True)
    n_groups = int(groups.max()) + 1
    totals_train = marginal_matrix(train, groups, n_groups).tocsc()
    totals_test = marginal_matrix(test, groups, n_groups).tocsc()
    selected_groups = groups[columns]
    sum_x = np.asarray(train[:, columns].sum(axis=0)).ravel()
    sum_z = np.asarray(totals_train.sum(axis=0)).ravel()
    sum_z2 = np.asarray(totals_train.power(2).sum(axis=0)).ravel()
    variance_z = sum_z2 - sum_z * sum_z / train.shape[0]
    coefficient = np.zeros(len(columns))

    for group in np.unique(selected_groups):
        positions = np.flatnonzero(selected_groups == group)
        selected = columns[positions]
        cross = (train[:, selected].T @ totals_train[:, group]).toarray().ravel()
        covariance = cross - sum_x[positions] * sum_z[group] / train.shape[0]
        if variance_z[group] > 0:
            coefficient[positions] = covariance / variance_z[group]

    intercept = (
        sum_x / train.shape[0]
        - coefficient * sum_z[selected_groups] / train.shape[0]
    )
    train_residual = (
        train[:, columns].toarray()
        - totals_train[:, selected_groups].toarray() * coefficient
        - intercept
    )
    test_residual = (
        test[:, columns].toarray()
        - totals_test[:, selected_groups].toarray() * coefficient
        - intercept
    )
    return sparse.csr_matrix(train_residual), sparse.csr_matrix(test_residual)
