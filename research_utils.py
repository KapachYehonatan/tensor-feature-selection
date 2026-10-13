"""Shared sparse calculations for the three MW/BH comparison methods."""
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent
SEED = MODEL_SEED = 42
MANN_WHITNEY_CHUNK = 2000
K_VALUES = sorted(set(range(1, 11)) | {20, 35, 50, 60, 75, 85, 100})
JOIN_PYTHON = Path("/home/kapachy/Microbium/bernoulli-correlation-join/.venv/bin/python")


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


def unique_day_zero_stool_mask(subjects, body_sites, days, reads, sample_ids):
    """Select one day-zero stool sample per subject, preferring read depth."""
    subjects = np.asarray(subjects, dtype=object)
    body_sites = np.asarray(body_sites, dtype=object)
    days = np.asarray(days, dtype=float)
    reads = np.asarray(reads, dtype=float)
    sample_ids = np.asarray(sample_ids, dtype=str)
    if len({len(subjects), len(body_sites), len(days), len(reads), len(sample_ids)}) != 1:
        raise ValueError("Baseline-selection columns must have equal lengths")

    selected = np.zeros(len(subjects), dtype=bool)
    best = {}
    for index in np.flatnonzero(
        np.array([isinstance(value, str) and bool(value) for value in subjects])
        & (np.char.lower(body_sites.astype(str)) == "stool")
        & (days == 0)
    ):
        subject = subjects[index]
        incumbent = best.get(subject)
        read_count = reads[index] if np.isfinite(reads[index]) else -np.inf
        if incumbent is None:
            best[subject] = index
            continue
        incumbent_reads = reads[incumbent] if np.isfinite(reads[incumbent]) else -np.inf
        if read_count > incumbent_reads or (
            read_count == incumbent_reads and sample_ids[index] < sample_ids[incumbent]
        ):
            best[subject] = index
    selected[list(best.values())] = True
    return selected


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


def new_model(model_seed=MODEL_SEED):
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=model_seed,
            solver="liblinear",
        ),
    )


def predict_model(train_x, train_y, test_x, model_seed=MODEL_SEED):
    if train_x.shape[1] == 0:
        return np.full(test_x.shape[0], np.mean(train_y))
    model = new_model(model_seed)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def collapsed_average(matrix, entity, total_entities):
    """Average retained pair columns belonging to each active entity."""
    counts = np.bincount(entity, minlength=total_entities)
    active = np.flatnonzero(counts)
    collapsed = marginal_matrix(matrix, entity, total_entities)[:, active]
    collapsed = collapsed @ sparse.diags(1.0 / counts[active])
    return sqrt_sparse(collapsed), active


def mann_whitney_many(matrix, targets, chunk_size=MANN_WHITNEY_CHUNK):
    """Compute two-sided MW statistics for the supplied target arrays."""
    matrix = sparse.csc_matrix(matrix)
    p_values = [np.ones(matrix.shape[1]) for _ in targets]
    real_effect = np.zeros(matrix.shape[1])
    for start in range(0, matrix.shape[1], chunk_size):
        end = min(start + chunk_size, matrix.shape[1])
        values = matrix[:, start:end].toarray()
        for target_number, target in enumerate(targets):
            positive, negative = target == 1, target == 0
            result = stats.mannwhitneyu(
                values[positive], values[negative], axis=0,
                alternative="two-sided", method="asymptotic",
            )
            p_values[target_number][start:end] = np.asarray(result.pvalue)
            if target_number == 0:
                denominator = positive.sum() * negative.sum()
                real_effect[start:end] = 2 * np.asarray(result.statistic) / denominator - 1
    return p_values, real_effect


def bh_family(matrix, targets):
    p_values, effect = mann_whitney_many(matrix, targets)
    real_q = benjamini_hochberg(p_values[0])
    columns = np.arange(matrix.shape[1])
    order = np.lexsort((columns, p_values[0], real_q))
    return {
        "order": order,
        "p": p_values[0],
        "q": real_q,
        "effect": effect,
    }


def feature_table(order, levels, ids, scores, maximum=None):
    if maximum is not None:
        order = order[:maximum]
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "level": np.asarray(levels)[order],
        "feature_id": np.asarray(ids)[order],
        "p": np.asarray(scores.get("p", np.full(len(levels), np.nan)))[order],
        "q": np.asarray(scores.get("q", np.full(len(levels), np.nan)))[order],
        "effect": np.asarray(scores.get("effect", np.full(len(levels), np.nan)))[order],
    })


def aggregate_pair_clusters(train, test, labels):
    """Average normalized original pair abundances within learned clusters."""
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (train.shape[1],) or train.shape[1] != test.shape[1]:
        raise ValueError("cluster labels must align with train and test columns")
    if np.any(labels < 0):
        raise ValueError("every retained feature must have a nonnegative cluster")
    cluster_count = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=cluster_count)
    assignment = sparse.csr_matrix(
        (1.0 / sizes[labels], (np.arange(len(labels)), labels)),
        shape=(len(labels), cluster_count),
    )
    return sqrt_sparse(train @ assignment), sqrt_sparse(test @ assignment), sizes


def screen(train, test, y, ids, level, sizes):
    scores = bh_family(train, [y])
    order = scores["order"][scores["q"][scores["order"]] <= .10]
    table = feature_table(order, np.full(len(ids), level), ids, scores)
    table["pair_count"] = np.asarray(sizes)[order]
    return train[:, order].tocsr(), test[:, order].tocsr(), table


def evaluate(train, test, ytrain, ytest, ks, c=1):
    rows, predictions, cache = [], [], {}
    for k in ks:
        effective = min(k, train.shape[1])
        if effective not in cache:
            probability = np.full(len(ytest), ytrain.mean())
            if effective:
                model = new_model(SEED)
                model.set_params(logisticregression__C=c)
                model.fit(train[:, :effective], ytrain)
                probability = model.predict_proba(test[:, :effective])[:, 1]
            cache[effective] = probability
        probability = cache[effective]
        predicted = probability >= .5
        rows.append(dict(requested_k=k, effective_k=effective,
                         auroc=roc_auc_score(ytest, probability),
                         precision=precision_score(ytest, predicted, zero_division=0),
                         recall=recall_score(ytest, predicted, zero_division=0),
                         f1=f1_score(ytest, predicted, zero_division=0)))
        predictions.extend(dict(requested_k=k, test_position=i, y=int(ytest[i]),
                                probability=float(value)) for i, value in enumerate(probability))
    return pd.DataFrame(rows), pd.DataFrame(predictions)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_signature(marker, candidate):
    """Accept only the explicitly verified refactor, never arbitrary old caches."""
    marker = Path(marker)
    if not marker.exists():
        return candidate
    saved = json.loads(marker.read_text())
    previous = saved.get('signature', hashlib.sha256(
        json.dumps(saved, sort_keys=True).encode()).hexdigest())
    if previous == candidate:
        return candidate
    bridge_path = PROJECT/'logs/project_cleanup_compatibility.json'
    if not bridge_path.exists():
        return candidate
    bridge = json.loads(bridge_path.read_text())
    if not marker.resolve().is_relative_to(PROJECT):
        return candidate
    key = str(marker.resolve().relative_to(PROJECT))
    entry = bridge['entries'].get(key)
    if entry != dict(old_signature=previous, new_signature=candidate):
        return candidate
    for name, expected in bridge['code_hashes'].items():
        if not Path(name).is_file() or file_hash(name) != expected:
            raise ValueError(f'Refactored cache code changed: {name}')
    for name, size, mtime in bridge['input_states'] + bridge['cache_states']:
        path = Path(name)
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (size, mtime):
            raise ValueError(f'Validated cache input/artifact changed: {name}')
    return previous
