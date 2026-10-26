"""Fixed zero/low/high bins and exp(log-likelihood ratio), lambda=1."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp, xlogy
from methods.correction import EBH


def log_choose(n, k):
    return gammaln(np.asarray(n) + 1) - gammaln(np.asarray(k) + 1) - gammaln(np.asarray(n) - k + 1)


def log_lr(counts, positives, total_positive):
    counts, positives = np.asarray(counts), np.asarray(positives)
    total = counts.sum(axis=-1)
    fitted = (xlogy(positives, positives) + xlogy(counts - positives, counts - positives) - xlogy(counts, counts)).sum(axis=-1)
    null = xlogy(total_positive, total_positive) + xlogy(total - total_positive, total - total_positive) - xlogy(total, total)
    return np.maximum(fitted - null, 0.)


def prepare_bins(matrix):
    x = sparse.csc_matrix(matrix, copy=True, dtype=float)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError("Invalid supplied values")
    n, d = x.shape
    support = np.diff(x.indptr)
    bins = np.ones(x.nnz, np.int8)
    for j in range(d):
        a, b = x.indptr[j:j + 2]
        if b > a:
            bins[a:b] += x.data[a:b] > np.median(x.data[a:b])
    high = sparse.csc_matrix(((bins == 2).astype(np.int8), x.indices.copy(), x.indptr.copy()), shape=x.shape)
    high.eliminate_zeros()
    presence = x.copy()
    presence.data = np.ones(x.nnz, np.int8)
    high_count = np.asarray(high.sum(axis=0)).ravel().astype(int)
    return presence, high, np.column_stack((n - support, support - high_count, high_count))


def table(n0, nl, nh, p):
    counts = np.array([n0, nl, nh], int)
    cells = np.array([(p - a - b, a, b) for a in range(min(nl, p) + 1)
                      for b in range(min(nh, p) + 1) if 0 <= p - a - b <= n0])
    mass = log_choose(counts, cells).sum(axis=1) - log_choose(counts.sum(), p)
    if abs(float(logsumexp(mass))) > 1e-10:
        raise ValueError("Incomplete null support")
    return mass, log_lr(counts, cells, p)


@lru_cache(None)
def normalizer(n0, nl, nh, p):
    mass, score = table(n0, nl, nh, p)
    return float(logsumexp(mass + score))


def evidence(prepared, targets):
    presence, high_matrix, counts = prepared
    targets = np.asarray(targets)
    n = presence.shape[0]
    if (targets.ndim != 2 or not len(targets) or targets.shape[1] != n
            or not np.isin(targets, [0, 1]).all() or np.any(targets.sum(axis=1) != targets[0].sum())):
        raise ValueError("Expected aligned class-count-preserving labels")
    p = int(targets[0].sum())
    if p in (0, n):
        raise ValueError("Both classes required")
    if not len(counts):
        return np.empty((len(targets), 0))
    present = np.asarray(targets.astype(np.int64) @ presence)
    high = np.asarray(targets.astype(np.int64) @ high_matrix)
    cells = np.stack((p - present, present - high, high), axis=-1)
    if np.any(cells < 0) or np.any(cells > counts[None]):
        raise ValueError("Invalid observed cells")
    return log_lr(counts[None], cells, p) - np.array([normalizer(*map(int, row), p) for row in counts])


@dataclass(frozen=True)
class Bins:
    kind = "log_e"
    correction: EBH = field(default_factory=EBH)

    def test(self, matrix, y):
        return dict(log_e=evidence(prepare_bins(matrix), np.asarray(y)[None])[0])

    def batch_test(self, matrix, targets, prepared=None):
        return evidence(prepare_bins(matrix) if prepared is None else prepared, targets)

    def family_size(self, candidates, original):
        return original

    def prepare(self, matrix):
        return prepare_bins(matrix)

    def test_settings(self):
        return {k: v for k, v in asdict(self).items() if k != "correction"}

    def dependencies(self):
        return (type(self).test, type(self).batch_test, type(self).prepare, type(self).test_settings, prepare_bins,
                evidence, table, log_choose, log_lr, normalizer.__wrapped__)

    def selection_dependencies(self):
        return (type(self).select, type(self).family_size, type(self.correction).select)

    def select(self, result, matrix, ids, family_size, rank_limit):
        order = np.lexsort((ids, -result["log_e"]))
        return dict(order=order[:rank_limit], native=self.correction.select(result["log_e"], ids, family_size))
