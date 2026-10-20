"""Shuffled FDP test: one model-dependent, one-sided discovery-count p-value."""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from methods.selectors.mann_whitney import mann_whitney_many, multiple_testing

SEED = 42

def label_permutations(y, key, count, seed=SEED):
    """Paired across representations, preserving class counts."""
    y = np.asarray(y, dtype=np.int8)
    if count < 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Permutations require a positive count and two classes')
    identity = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
    return np.stack([rng.permutation(y) for _ in range(count)])


def permutation_discoveries(train, targets, alpha=.10):
    p_values, _ = mann_whitney_many(train, targets)
    rows = []
    for index, p in enumerate(p_values):
        for correction, q in multiple_testing(p).items():
            rows.append(dict(permutation=index, correction=correction,
                             discoveries=int(np.sum(q <= alpha))))
    return pd.DataFrame(rows)


def _discovery_counts(discoveries, n):
    values = np.asarray(discoveries, dtype=float)
    if (not np.isscalar(n) or not np.isfinite(n) or n < 0 or int(n) != n or
            values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or
            np.any(values < 0) or np.any(values > n) or np.any(values != np.floor(values))):
        raise ValueError('Invalid tested-family size or permutation discovery counts')
    return values.astype(np.int64)


def discovery_statistics(discoveries, r, n):
    """Requires the pooled binomial/normal approximation; not an FDR guarantee."""
    values = _discovery_counts(discoveries, n)
    if not np.isfinite(r) or r < 0 or r > n or int(r) != r:
        raise ValueError('Invalid observed discovery count')
    m, mean = len(values), float(values.mean())
    pooled_mean = (r+values.sum())/(m+1)
    pooled_p = pooled_mean/n if n else 0.
    variance = pooled_mean*(1-pooled_p)*(1+1/m)
    expected = [pooled_mean, n-pooled_mean, m*pooled_mean, m*(n-pooled_mean)]
    status = ('empty_family' if n == 0 else 'all_zero' if pooled_mean == 0 else
              'all_positive' if pooled_mean == n else 'nondegenerate')
    z = (r-mean)/np.sqrt(variance) if variance > 0 else 0.
    log_p = float(stats.norm.logsf(z)) if variance > 0 else 0.
    p = float(np.exp(log_p))
    return dict(tested_features=int(n), real_discoveries=int(r), permutations=m,
        mean_shuffled_discoveries=mean, small_expected_counts=bool(min(expected) < 5),
        status=status, p_value=p if p else np.nan, log_p_value=log_p, p_value_underflow=p == 0)
