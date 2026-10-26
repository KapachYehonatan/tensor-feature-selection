"""Discovery-count permutation p-value and its historical normal reference."""
import hashlib
import numpy as np
from scipy import stats


def label_permutations(y, key, count, seed=42):
    y = np.asarray(y, dtype=np.int8)
    if count < 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Shuffles require a positive budget and two classes")
    identity = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
    return np.stack([rng.permutation(y) for _ in range(count)])


def discovery_statistics(counts, real, family_size):
    values = np.asarray(counts, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not np.isfinite(real) or not np.isfinite(family_size)
            or family_size < 0 or int(family_size) != family_size
            or real < 0 or real > family_size or int(real) != real
            or np.any(values < 0) or np.any(values > family_size) or np.any(values != np.floor(values))):
        raise ValueError("Invalid complete discovery counts")
    m, mean = len(values), float(values.mean())
    exceedances = int(np.count_nonzero(values >= real))
    p = (exceedances + 1) / (m + 1)
    pooled_mean = (real + values.sum()) / (m + 1)
    pooled_p = pooled_mean / family_size if family_size else 0
    variance = pooled_mean * (1 - pooled_p) * (1 + 1 / m)
    z = (real - mean) / np.sqrt(variance) if variance > 0 else 0.
    normal_log_p = float(stats.norm.logsf(z)) if variance > 0 else 0.
    return dict(tested_features=int(family_size), real_discoveries=int(real), permutations=m,
                exceedances=exceedances, mean_shuffled_discoveries=mean,
                p_value=p, log_p_value=float(np.log(p)), normal_log_p_value=normal_log_p)
