"""Unscreened MW ranking and separate complete-family BH/BY discoveries."""
import numpy as np
import pandas as pd
from scipy import sparse, stats

MANN_WHITNEY_CHUNK = 2000

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


def multiple_testing(p):
    """Correct the complete eligible family, independently of prediction Top-K."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Multiple testing requires finite p-values in [0, 1]')
    return {name: stats.false_discovery_control(p, method=name) if p.size else p.copy()
            for name in ('bh', 'by')}


def mw_ranked_family(train, y, ids, level, sizes, alpha=.10):
    """Label-blind constant exclusion, then unfiltered p/ID ranking and BH/BY."""
    train = sparse.csc_matrix(train)
    eligible = (np.flatnonzero(train.max(axis=0).toarray().ravel() >
                              train.min(axis=0).toarray().ravel())
                if train.shape[1] else np.array([], dtype=int))
    p, effect = mann_whitney_many(train[:, eligible], [np.asarray(y)])
    adjusted = multiple_testing(p[0])
    order = np.lexsort((np.asarray(ids)[eligible], p[0]))
    columns = eligible[order]
    table = pd.DataFrame(dict(rank=np.arange(1, len(columns)+1), level=level,
        feature_id=np.asarray(ids)[columns], pair_count=np.asarray(sizes)[columns],
        p=p[0][order], effect=effect[order]))
    for correction in ('bh', 'by'):
        table['q_'+correction] = adjusted[correction][order]
        table['significant_'+correction] = table['q_'+correction].le(alpha)
    return columns, table

