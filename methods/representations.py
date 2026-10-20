"""Mean supplied values over training-defined feature groups."""
import numpy as np
from scipy import sparse

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


def collapsed_average(matrix, entity, total_entities):
    """Average retained pair columns belonging to each active entity."""
    counts = np.bincount(entity, minlength=total_entities)
    active = np.flatnonzero(counts)
    collapsed = marginal_matrix(matrix, entity, total_entities)[:, active]
    collapsed = collapsed @ sparse.diags(1.0 / counts[active])
    return collapsed.tocsr(), active


def aggregate_pair_clusters(train, test, labels):
    """Average supplied pair values within learned clusters, including zeros."""
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
    return (train @ assignment).tocsr(), (test @ assignment).tocsr(), sizes

