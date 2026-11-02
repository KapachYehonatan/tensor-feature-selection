"""Mean supplied values over training-defined feature groups."""
import numpy as np
from scipy import sparse

def marginal_matrix(matrix: sparse.spmatrix, pair_entity: np.ndarray, n_entities: int) -> sparse.csr_matrix:
    """Sum sample×pair values into sample×entity columns using one entity ID per pair."""
    pair_entity = np.asarray(pair_entity, dtype=np.int64)
    if matrix.shape[1] != len(pair_entity):
        raise ValueError("Pair mapping does not match matrix columns")
    mapping = sparse.csr_matrix(
        (np.ones(len(pair_entity)), (np.arange(len(pair_entity)), pair_entity)),
        shape=(len(pair_entity), n_entities),
    )
    return (matrix @ mapping).tocsr()


def collapsed_average(matrix: sparse.spmatrix, entity: np.ndarray,
                      total_entities: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return sample×active-entity means and their original entity IDs.

    Every retained pair contributes to its group's denominator, including zeros.
    """
    counts = np.bincount(entity, minlength=total_entities)
    active = np.flatnonzero(counts)
    collapsed = marginal_matrix(matrix, entity, total_entities)[:, active]
    collapsed = collapsed @ sparse.diags(1.0 / counts[active])
    return collapsed.tocsr(), active


def aggregate_pair_clusters(matrix: sparse.spmatrix, labels: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return sample×cluster means and cluster sizes from one label per retained pair."""
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (matrix.shape[1],):
        raise ValueError("cluster labels must align with pair columns")
    if np.any(labels < 0):
        raise ValueError("every retained feature must have a nonnegative cluster")
    cluster_count = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=cluster_count)
    assignment = sparse.csr_matrix(
        (1.0 / sizes[labels], (np.arange(len(labels)), labels)),
        shape=(len(labels), cluster_count),
    )
    return (matrix @ assignment).tocsr(), sizes

