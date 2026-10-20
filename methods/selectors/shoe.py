"""Standalone SHOE: honest histogram learning, one evidence block, native e-BH."""
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import logit, gammaln, logsumexp
from methods.preprocessing import prevalence_mask, retained_pair_values

SEED, ALPHA = 42, .10


def histogram_scores(membership, bin_counts, labels):
    """Laplace smoothing already makes every log-odds finite; no clipping knobs."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != membership.shape[0] or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid score-training labels')
    sizes = np.asarray(bin_counts).reshape(-1, 2)
    positive = np.asarray(labels.astype(np.int64) @ membership).reshape(len(labels), -1, 2)
    zero_size = membership.shape[0]-sizes.sum(axis=1)
    zero_positive = labels.sum(axis=1)[:, None]-positive.sum(axis=2)
    q0, q = (zero_positive+1)/(zero_size+2), (positive+1)/(sizes+2)
    if not (np.all((q > 0)&(q < 1)) and np.all((q0 > 0)&(q0 < 1))):
        raise ValueError('Invalid smoothed histogram counts')
    return logit(q)-logit(q0)[:, :, None]


def transform(matrix, fitted):
    """Select training-defined columns without changing supplied values."""
    matrix = sparse.csr_matrix(matrix, dtype=float, copy=True)
    matrix.sum_duplicates(); matrix.eliminate_zeros()
    if matrix.shape[1] != fitted['input_features'] or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Invalid abundance matrix or changed raw pair columns')
    retained = fitted['retained_pair_ids']
    if not len(retained):
        return sparse.csr_matrix((matrix.shape[0], 0))
    return retained_pair_values(matrix, retained)[:, fitted['eligible_positions']]


def fit(matrix, labels, key, alpha=ALPHA):
    """Fit on training samples only; a separate validation matrix uses transform().

    Input columns are raw bacterium–gene pairs. Returned order/native arrays
    index the returned features matrix; feature_ids map those columns to raw IDs.
    """
    matrix = sparse.csr_matrix(matrix, dtype=float, copy=True)
    matrix.sum_duplicates(); matrix.eliminate_zeros()
    labels = np.asarray(labels)
    if (labels.shape != (matrix.shape[0],) or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
            or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)):
        raise ValueError('Invalid training abundances or binary labels')
    training, evidence = honest_split(np.arange(len(labels)), key)
    p = prepare(matrix, training, evidence)
    scores = histogram_scores(p['membership'], p['bin_counts'], labels[None, training])
    log_e = log_evidence(p['evidence_bins'], scores, labels[None, evidence])[0]
    order, native, weighted = native_selection(log_e, p['feature_ids'], alpha)
    return dict(input_features=matrix.shape[1], retained_pair_ids=p['retained_pair_ids'],
        eligible_positions=np.searchsorted(p['retained_pair_ids'], p['feature_ids']),
        feature_ids=p['feature_ids'], training_indices=training, evidence_indices=evidence,
        prevalence_cutoff=p['prevalence_cutoff'], positive_median=p['median'],
        one_positive_bin=p['one_bin'], contrasts=scores[0], log_evidence=log_e,
        log_weighted_evidence=weighted, order=order, native=native,
        selected_pair_ids=p['feature_ids'][native], features=p['features'])


def honest_split(indices, key):
    identity = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, identity]))
    order = rng.permutation(np.asarray(indices, dtype=np.int64))
    if len(order) < 2: raise ValueError('Score learning and evidence need disjoint nonempty samples')
    return order[:len(order)//2], order[len(order)//2:]


def native_selection(log_e, feature_ids, alpha=ALPHA):
    log_e, feature_ids = np.asarray(log_e), np.asarray(feature_ids)
    if (log_e.ndim != 1 or feature_ids.shape != log_e.shape or len(np.unique(feature_ids)) != len(feature_ids)
            or np.isnan(log_e).any() or np.isposinf(log_e).any() or not 0 < alpha < 1):
        raise ValueError('Invalid evidence, feature identities or FDR level')
    d = len(log_e)
    log_a = log_e-np.log(d) if d else log_e.copy()
    order = np.lexsort((feature_ids, -log_a))
    valid = np.flatnonzero(log_a[order] >= -np.log(alpha)-np.log(np.arange(1, d+1)))
    discoveries = int(valid[-1]+1) if len(valid) else 0
    return order, order[:discoveries], log_a


def prepare(raw, fit_indices, evidence_indices):
    """Fit preprocessing on score-training X only; transform rows independently."""
    raw = sparse.csr_matrix(raw, dtype=float, copy=True)
    raw.sum_duplicates(); raw.eliminate_zeros()
    fit_indices, evidence_indices = map(np.asarray, (fit_indices, evidence_indices))
    if (not len(fit_indices) or not len(evidence_indices) or
            len(np.unique(np.r_[fit_indices, evidence_indices])) != len(fit_indices)+len(evidence_indices) or
            np.any(np.r_[fit_indices, evidence_indices] < 0) or
            np.any(np.r_[fit_indices, evidence_indices] >= raw.shape[0]) or
            not np.isfinite(raw.data).all() or np.any(raw.data < 0)):
        raise ValueError('Invalid abundance matrix or overlapping internal samples')
    mask, _, cutoff = prevalence_mask(raw[fit_indices], .05, 5)
    retained = np.flatnonzero(mask)
    features = retained_pair_values(raw, retained)
    t = features[fit_indices].tocsc()
    eligible = (np.flatnonzero(t.max(axis=0).toarray().ravel() > t.min(axis=0).toarray().ravel())
                if t.shape[1] else np.array([], dtype=int))
    features = features[:, eligible].tocsr()
    t, v = features[fit_indices].tocsc(), features[evidence_indices].tocsc()
    t.sort_indices(); v.sort_indices()
    d = features.shape[1]
    median, one_bin = np.zeros(d), np.zeros(d, bool)
    tb, vb = np.zeros(t.nnz, np.uint8), np.zeros(v.nnz, np.uint8)
    for j in range(d):
        ta, tz, va, vz = t.indptr[j], t.indptr[j+1], v.indptr[j], v.indptr[j+1]
        values = t.data[ta:tz]
        median[j] = np.median(values)
        one_bin[j] = not np.any(values > median[j])
        tb[ta:tz] = 1+(values > median[j]) if not one_bin[j] else 1
        vb[va:vz] = 1+(v.data[va:vz] > median[j]) if not one_bin[j] else 1
    tc = np.repeat(np.arange(d), np.diff(t.indptr))
    membership = sparse.csr_matrix((np.ones(t.nnz, np.int8), (t.indices, 2*tc+tb-1)),
                                   shape=(len(fit_indices), 2*d))
    bins = sparse.csc_matrix((vb, v.indices.copy(), v.indptr.copy()), shape=v.shape)
    counts = np.asarray(membership.sum(axis=0)).ravel().astype(np.int64)
    return dict(features=features, membership=membership, evidence_bins=bins,
        feature_ids=retained[eligible], retained_pair_ids=retained, median=median, one_bin=one_bin,
        bin_counts=counts, fit_indices=fit_indices, evidence_indices=evidence_indices,
        prevalence_cutoff=cutoff)


WORK_ELEMENTS = 500_000  # Computational batch size only; does not change the statistic.


def log_evidence(bins, contrasts, labels):
    """Exact full-orbit evidence for frozen zero/low/high histogram scores.

    Only the total evidence-half class count is conditioned on. The denominator
    sums over feasible allocations of positive labels to the three bins, with
    their binomial multiplicities; it does not enumerate individual labelings.
    """
    bins = sparse.csc_matrix(bins)
    labels, contrasts = np.asarray(labels), np.asarray(contrasts)
    m, d = bins.shape
    if (labels.ndim != 2 or labels.shape[1] != m or contrasts.shape != (len(labels), d, 2)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(contrasts).all()
            or not np.isin(bins.data, [1, 2]).all() or not bins.has_canonical_format):
        raise ValueError('Invalid or misaligned frozen bins, scores or evidence labels')
    support = np.diff(bins.indptr)
    high = np.asarray(bins.sum(axis=0)).ravel().astype(int)-support
    low = support-high
    present = bins.copy()
    present.data = np.ones(present.nnz, np.int8)
    total_positive = np.asarray(labels.astype(np.int64) @ present)
    high_positive = np.asarray(labels.astype(np.int64) @ bins)-total_positive
    low_positive = total_positive-high_positive
    out = np.full((len(labels), d), -np.inf)
    if not d or not m:
        return out
    factorial = gammaln(np.arange(m+1)+1)
    k = np.arange(m+1)
    with np.errstate(invalid='ignore'):
        log_choose = factorial[:, None]-factorial[k][None, :]-factorial[np.maximum(k[:, None]-k, 0)]
    log_choose[k[None, :] > k[:, None]] = -np.inf
    keys = low*(m+1)+high
    order = np.argsort(keys, kind='stable')
    edges = np.r_[0, 1+np.flatnonzero(np.diff(keys[order])), d]
    for b, y in enumerate(labels):
        positives = int(y.sum())
        if positives in (0, m):
            out[b, support > 0] = 0.
            continue
        r = contrasts[b]
        observed = low_positive[b]*r[:, 0]+high_positive[b]*r[:, 1]
        for start, end in zip(edges[:-1], edges[1:]):
            columns = order[start:end]
            n1, n2 = int(low[columns[0]]), int(high[columns[0]])
            n0 = m-n1-n2
            if n1+n2 == 0:
                continue
            a, c = np.meshgrid(np.arange(min(n1, positives)+1),
                               np.arange(min(n2, positives)+1), indexing='ij')
            z = positives-a-c
            feasible = (z >= 0)&(z <= n0)
            a, c, z = a[feasible], c[feasible], z[feasible]
            log_mass = (log_choose[n1, a]+log_choose[n2, c]+log_choose[n0, z]
                        -log_choose[m, positives])
            chunk = max(1, WORK_ELEMENTS//len(a))
            for left in range(0, len(columns), chunk):
                ids = columns[left:left+chunk]
                normalizer = logsumexp(log_mass[:, None]+a[:, None]*r[ids, 0]
                                       +c[:, None]*r[ids, 1], axis=0)
                out[b, ids] = observed[ids]-normalizer
    if np.isnan(out).any() or np.isposinf(out).any():
        raise ValueError('Nonfinite full-orbit evidence')
    return out


def fit_evidence(prepared, targets):
    """Refit score probabilities for each real or shuffled label vector."""
    contrasts = histogram_scores(prepared['membership'], prepared['bin_counts'],
                                  np.asarray(targets)[:, prepared['fit_indices']])
    evidence = log_evidence(prepared['evidence_bins'], contrasts,
                           np.asarray(targets)[:, prepared['evidence_indices']])
    return evidence, contrasts
