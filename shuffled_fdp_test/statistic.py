"""Upper-tail discovery-count permutation test; exchangeable labels, not FDR control."""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from methods.selectors.mann_whitney import mann_whitney_many

SEED = 42

def label_permutations(y, key, count, seed=SEED):
    """Paired across representations, preserving class counts."""
    y = np.asarray(y, dtype=np.int8)
    if count < 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Permutations require a positive count and two classes')
    identity = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
    return np.stack([rng.permutation(y) for _ in range(count)])


def permutation_discoveries(train, targets, alpha=.10, corrections=('bh','by')):
    p_values, _ = mann_whitney_many(train, targets)
    rows = []
    for index, p in enumerate(p_values):
        for correction in corrections:
            q = stats.false_discovery_control(p, method=correction) if p.size else p
            rows.append(dict(permutation=index, correction=correction,
                             discoveries=int(np.sum(q <= alpha))))
    return pd.DataFrame(rows)


def _discovery_counts(discoveries, n):
    values = np.asarray(discoveries, dtype=float)
    if (not np.isscalar(n) or not np.isfinite(n) or n < 0 or int(n) != n or
            values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all() or
            np.any(values < 0) or np.any(values > n) or np.any(values != np.floor(values))):
        raise ValueError('Invalid tested-family size or permutation discovery counts')
    return values.astype(np.int64)


def discovery_statistics(discoveries, r, n):
    """H0: label exchangeability; H1: a count shifted upward. Ties are conservative."""
    values = _discovery_counts(discoveries, n)
    if not np.isfinite(r) or r < 0 or r > n or int(r) != r:
        raise ValueError('Invalid observed discovery count')
    m, mean = len(values), float(values.mean())
    exceedances = int(np.count_nonzero(values >= r))
    p = (exceedances+1)/(m+1)
    status = ('empty_family' if n == 0 else 'all_zero' if r == 0 and not values.any() else
              'all_positive' if r == n and np.all(values == n) else 'nondegenerate')
    return dict(tested_features=int(n), real_discoveries=int(r), permutations=m,
        mean_shuffled_discoveries=mean, exceedances=exceedances, p_value=p,
        log_p_value=float(np.log(p)), p_value_method='permutation', status=status)


def normal_approximation_statistics(discoveries, r, n):
    """Previous binomial/normal count approximation, retained only for comparison."""
    values = _discovery_counts(discoveries, n)
    if not np.isfinite(r) or r < 0 or r > n or int(r) != r:
        raise ValueError('Invalid observed discovery count')
    m, mean = len(values), float(values.mean())
    pooled_mean = (r+values.sum())/(m+1)
    pooled_p = pooled_mean/n if n else 0
    variance = pooled_mean*(1-pooled_p)*(1+1/m)
    z = (r-mean)/np.sqrt(variance) if variance > 0 else 0.
    log_p = float(stats.norm.logsf(z)) if variance > 0 else 0.
    p = float(np.exp(log_p))
    expected = [pooled_mean,n-pooled_mean,m*pooled_mean,m*(n-pooled_mean)]
    status = ('empty_family' if n == 0 else 'all_zero' if pooled_mean == 0 else
              'all_positive' if pooled_mean == n else 'nondegenerate')
    return dict(p_value=p if p else np.nan,log_p_value=log_p,p_value_underflow=p == 0,
                small_expected_counts=min(expected)<5,status=status)
