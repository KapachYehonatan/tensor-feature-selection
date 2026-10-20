"""Raw-pair MW and exact conditional Hurdle evidence; no score-learning split."""
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp
from scipy.stats import hypergeom
from methods.preprocessing import prevalence_mask, retained_pair_values
from methods.selectors.mann_whitney import mann_whitney_many
from experiments.cache import relocated_signature

GAMMA = .50
FEATURE_BATCH = 1024  # Bounds temporary coefficient arrays, not the candidate family.


def prepare(raw, training):
    """Freeze raw-pair columns on the entire training set; keep values unchanged."""
    raw = sparse.csr_matrix(raw, dtype=float, copy=True)
    raw.sum_duplicates(); raw.eliminate_zeros()
    training = np.asarray(training)
    if (training.ndim != 1 or not len(training) or len(np.unique(training)) != len(training)
            or not np.issubdtype(training.dtype, np.integer) or np.any(training < 0)
            or np.any(training >= raw.shape[0]) or not np.isfinite(raw.data).all()
            or np.any(raw.data < 0) or np.any(np.asarray(raw.sum(axis=1)).ravel() <= 0)):
        raise ValueError('Invalid training indices or supplied pair values')
    mask, _, cutoff = prevalence_mask(raw[training], .05, 5)
    retained = np.flatnonzero(mask)
    features = retained_pair_values(raw, retained)
    t = features[training]
    eligible = (np.flatnonzero(t.max(axis=0).toarray().ravel() > t.min(axis=0).toarray().ravel())
                if t.shape[1] else np.array([], int))
    return dict(features=features[:, eligible].tocsr(), feature_ids=retained[eligible],
                retained_pair_ids=retained, fit_indices=training, prevalence_cutoff=cutoff)


def mw_log_evidence(matrix, targets):
    """Approximate MW p-values converted with fixed gamma=.5, without BH/BY."""
    p, effect = mann_whitney_many(matrix, np.asarray(targets))
    p = np.asarray(p)
    if np.any(p <= 0) or np.any(p > 1) or not np.isfinite(p).all():
        raise ValueError('Invalid/underflowed MW p-value; refusing infinite evidence')
    return np.log1p(-GAMMA)-GAMMA*np.log(p), p, effect


def log_coefficients(log_weights, degree):
    """Log elementary symmetric coefficient, batched over leading dimensions."""
    result = np.full((*log_weights.shape[:-1], degree+1), -np.inf)
    result[..., 0] = 0.
    for i in range(log_weights.shape[-1]):
        end = min(degree, i+1)
        result[..., 1:end+1] = np.logaddexp(
            result[..., 1:end+1], result[..., :end]+log_weights[..., i, None])
    return result[..., degree]


def hurdle_components(matrix, targets, cache_dir=None):
    """Exact two-sided components, sharing normalizers over label permutations.

    Standardizing the statistic inside its exponential changes no stored values.
    Cache completed feature blocks, including all supplied label vectors, so an
    interrupted permutation analysis does not repeat its expensive recursions.
    """
    x = sparse.csc_matrix(matrix, dtype=float, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    y = np.asarray(targets)
    n, d = x.shape
    if (y.ndim != 2 or y.shape[1] != n or not np.isin(y, [0, 1]).all()
            or not np.isfinite(x.data).all() or np.any(x.data < 0)):
        raise ValueError('Invalid Hurdle values or labels')
    presence = x.copy(); presence.data = np.ones(x.nnz, np.int8)
    support = np.diff(x.indptr)
    k = np.asarray(y.astype(np.int64) @ presence)
    sums = np.asarray(y @ x)
    z, a = np.zeros((len(y), d)), np.zeros((len(y), d))
    lambdas, variances = np.zeros((len(y), d)), np.zeros(d)
    eta = np.zeros((len(y), d))
    signature = hashlib.sha256()
    for array in (x.data, x.indices, x.indptr, y, np.asarray(x.shape)):
        signature.update(np.ascontiguousarray(array).tobytes())
    signature.update(Path(__file__).read_bytes())
    identity = relocated_signature(cache_dir, signature.hexdigest())
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    for s in np.unique(support):
        columns = np.flatnonzero(support == s)
        if s == 0:
            continue
        for b, n1 in enumerate(y.sum(axis=1).astype(int)):
            variance = s*(n1/n)*(1-n1/n)*(n-s)/(n-1) if n > 1 else 0.
            if variance > 0:
                lam = 1/np.sqrt(variance)
                grid = np.arange(max(0, s+n1-n), min(s, n1)+1)
                mass = hypergeom.logpmf(grid, n, n1, int(s))
                plus = logsumexp(mass+lam*grid)
                minus = logsumexp(mass-lam*grid)
                z[b, columns] = np.logaddexp(lam*k[b, columns]-plus,
                                             -lam*k[b, columns]-minus)-np.log(2)
                lambdas[b, columns] = lam
        for left in range(0, len(columns), FEATURE_BATCH):
            ids = columns[left:left+FEATURE_BATCH]
            path = cache_dir/f's{s}_b{left}.npz' if cache_dir is not None else None
            if path is not None and path.exists():
                with np.load(path) as saved:
                    if str(saved['signature']) != identity or not np.array_equal(saved['columns'], ids):
                        raise ValueError(f'Changed Hurdle checkpoint: {path}')
                    a[:, ids], eta[:, ids], variances[ids] = saved['log_abundance'], saved['eta'], saved['variance']
                if (not np.isfinite(a[:, ids]).all() or not np.isfinite(eta[:, ids]).all()
                        or not np.isfinite(variances[ids]).all() or np.any(variances[ids] < 0)
                        or np.any(eta[:, ids] < 0)):
                    raise ValueError(f'Invalid Hurdle checkpoint: {path}')
                continue
            values = x.data[x.indptr[ids, None]+np.arange(s)]
            mean = values.mean(axis=1)
            centered = values-mean[:, None]
            variance = np.sum(centered*centered, axis=1)/(s-1) if s > 1 else np.zeros(len(ids))
            variances[ids] = variance
            sd = np.sqrt(variance)
            scaled = np.divide(centered, sd[:, None], out=np.zeros_like(centered), where=sd[:, None] > 0)
            h = np.minimum(k[:, ids], s-k[:, ids])
            observed = np.divide(sums[:, ids]-k[:, ids]*mean, sd,
                                 out=np.zeros((len(y), len(ids))), where=sd > 0)
            observed = np.where(k[:, ids] <= s/2, observed, -observed)
            for degree in np.unique(h):
                if degree == 0:
                    continue
                local = np.flatnonzero(np.any(h == degree, axis=0) & (sd > 0))
                if not len(local):
                    continue
                theta = np.sqrt(s/(degree*(s-degree)))
                weights = theta*scaled[local]
                normalizers = log_coefficients(np.stack((weights, -weights)), int(degree))
                log_choose = gammaln(s+1)-gammaln(degree+1)-gammaln(s-degree+1)
                normalizers -= log_choose
                log_a = np.logaddexp(theta*observed[:, local]-normalizers[0],
                                     -theta*observed[:, local]-normalizers[1])-np.log(2)
                active = h[:, local] == degree
                a[:, ids[local]] = np.where(active, log_a, a[:, ids[local]])
                eta[:, ids[local]] = np.where(active, 1/(sd[local]*theta), eta[:, ids[local]])
            if path is not None:
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, signature=identity, columns=ids,
                                    log_abundance=a[:, ids], eta=eta[:, ids], variance=variance)
                temporary.replace(path)
                if left == 0 or left+FEATURE_BATCH >= len(columns):
                    print(f'Hurdle support={s}: {min(left+FEATURE_BATCH,len(columns)):,}/{len(columns):,} features; {len(y)} label vectors', flush=True)
    if not all(np.isfinite(value).all() for value in [z, a, lambdas, eta, variances]):
        raise ValueError('Nonfinite Hurdle evidence')
    return dict(log_presence=z, log_abundance=a, lambda_=lambdas, eta=eta,
                support=support, present_variance=variances)


def combine(components):
    z, a = components['log_presence'], components['log_abundance']
    return dict(hurdle_product=z+a, hurdle_average=np.logaddexp(z, a)-np.log(2))
