"""Complete-discovery-set redundancy; no Top-K curve computation."""
import numpy as np

PAIRWISE_FEATURE_LIMIT = 10_000

def average_pairwise_absolute_correlation(matrix, columns):
    """Exact mean over distinct feature pairs; guard before any dense allocation."""
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

