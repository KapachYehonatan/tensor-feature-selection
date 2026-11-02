"""Exact pairwise redundancy for complete discovery sets or ranked Top-K subsets."""
import numpy as np
from scipy import sparse

PAIRWISE_FEATURE_LIMIT = 10_000

def average_pairwise_absolute_correlation(matrix: sparse.spmatrix, columns: np.ndarray) -> dict:
    """Evaluate selected columns of a sample×feature matrix; return value, size and status.

    Exclude self-correlations. Check the feature-count guard before densifying.
    """
    k = len(columns)
    result = dict(effective_k=k, pair_count=k*(k-1)//2,
                  average_pairwise_absolute_correlation=np.nan)
    if k < 2:
        return dict(result, status='undefined_subset_size')
    if k > PAIRWISE_FEATURE_LIMIT:
        return dict(result, status='skipped_size_limit')
    values = matrix[:, list(columns)].toarray()
    if not np.isfinite(values).all() or np.any(np.ptp(values, axis=0) == 0):
        raise ValueError('Constant/nonfinite selected feature in pairwise correlation')
    correlation = np.corrcoef(values, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError('Nonfinite pairwise correlation')
    np.abs(correlation, out=correlation)
    result['average_pairwise_absolute_correlation'] = float(
        (correlation.sum()-np.trace(correlation))/(k*(k-1)))
    return dict(result, status='defined')

