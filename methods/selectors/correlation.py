"""Greedy training-only pairwise filtering; no full neighbour search."""
import numpy as np
from scipy import sparse


def greedy_filter(matrix, order, threshold, limit=None):
    x = sparse.csc_matrix(matrix, dtype=float)
    order = np.asarray(order, dtype=int)
    if (not np.isfinite(threshold) or not 0 < threshold < 1 or order.ndim != 1
            or len(np.unique(order)) != len(order) or np.any(order < 0) or np.any(order >= x.shape[1])
            or (limit is not None and limit < 0)):
        raise ValueError("Invalid correlation-filter candidates")
    maximum = len(order) if limit is None else min(limit, len(order))
    directions = np.empty((maximum, x.shape[0]))
    accepted = []
    for column in order:
        if len(accepted) == maximum:
            break
        z = x[:, column].toarray().ravel()
        z -= z.mean()
        norm = np.linalg.norm(z)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Constant/nonfinite candidate")
        z /= norm
        if accepted and np.any(np.abs(directions[:len(accepted)] @ z) >= threshold):
            continue
        directions[len(accepted)] = z
        accepted.append(column)
    return np.asarray(accepted, dtype=int)
