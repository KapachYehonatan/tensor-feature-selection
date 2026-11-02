"""Unscreened MW ranking and separate complete-family discoveries."""
import numpy as np
from scipy import sparse, stats
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
from methods.correction import BY
from methods.selectors.correlation import greedy_filter

MANN_WHITNEY_CHUNK = 2000
SHUFFLE_WORKERS = 24


@dataclass
class PreparedMannWhitney:
    ranks: sparse.csc_matrix
    zero_ranks: np.ndarray
    tie_terms: np.ndarray
    workers: int

    def test(self, targets):
        targets = np.asarray(targets)
        n, p = self.ranks.shape
        if (targets.ndim != 2 or targets.shape[1] != n
                or not np.isin(targets, [0, 1]).all()):
            raise ValueError('Shuffle labels must be binary and match the prepared samples')
        groups = (targets == 1).astype(np.float64)
        n1 = groups.sum(axis=1)[:, None]
        n0 = n - n1
        if np.any((n1 == 0) | (n0 == 0)):
            raise ValueError('Mann-Whitney requires both label groups')
        result = np.empty((len(targets), p))

        def calculate(bounds):
            start, end = bounds
            rank_sum = groups @ self.ranks[:, start:end]
            rank_sum += n1 * self.zero_ranks[start:end]
            u = rank_sum - n1 * (n1 + 1) / 2
            variance = n1 * n0 / 12 * ((n + 1) - self.tie_terms[start:end] / (n * (n - 1)))
            numerator = u - n1 * n0 / 2
            numerator -= .5 * np.sign(numerator)
            result[:, start:end] = np.clip(2 * stats.norm.sf(np.abs(numerator / np.sqrt(variance))), 0, 1)

        width = max(MANN_WHITNEY_CHUNK, (p + self.workers - 1) // self.workers)
        bounds = [(start, min(start + width, p)) for start in range(0, p, width)]
        if self.workers == 1:
            for block in bounds:
                calculate(block)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(calculate, bounds))
        return result


def prepare_mann_whitney(matrix, workers=SHUFFLE_WORKERS):
    """Pool ranks once; implicit zeros share an offset, avoiding a dense rank matrix."""
    if not isinstance(workers, int) or not 1 <= workers <= SHUFFLE_WORKERS:
        raise ValueError('Use between one and 24 shared-memory Mann-Whitney workers')
    matrix = sparse.csc_matrix(matrix, copy=True)
    if matrix.dtype != np.float64:
        raise ValueError('Prepared Mann-Whitney requires float64 predictors to preserve SciPy arithmetic')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    n, p = matrix.shape
    if n < 2 or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Prepared Mann-Whitney requires finite nonnegative values and at least two samples')
    zero_ranks = (n - np.diff(matrix.indptr) + 1) / 2
    rank_values = np.empty(matrix.nnz, dtype=np.float64)
    tie_terms = np.empty(p)

    def prepare(bounds):
        start, end = bounds
        values = matrix[:, start:end].toarray()
        ranks = stats.rankdata(values, axis=0) - zero_ranks[start:end]
        low, high = matrix.indptr[start], matrix.indptr[end]
        columns = np.repeat(np.arange(end - start), np.diff(matrix.indptr[start:end + 1]))
        rank_values[low:high] = ranks[matrix.indices[low:high], columns]
        ordered = np.sort(values, axis=0)
        boundaries = np.ones((n + 1, end - start), dtype=bool)
        boundaries[1:n] = ordered[1:] != ordered[:-1]
        feature, positions = np.nonzero(boundaries.T)
        same = feature[1:] == feature[:-1]
        lengths = np.diff(positions)[same].astype(np.float64)
        tie_terms[start:end] = np.bincount(feature[:-1][same], lengths**3 - lengths, minlength=end - start)

    bounds = [(start, min(start + MANN_WHITNEY_CHUNK, p)) for start in range(0, p, MANN_WHITNEY_CHUNK)]
    if workers == 1:
        for block in bounds:
            prepare(block)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(prepare, bounds))
    if np.any(tie_terms >= n**3 - n):
        raise ValueError('Training-constant predictors must be excluded before Mann-Whitney shuffles')
    ranks = sparse.csc_matrix((rank_values, matrix.indices, matrix.indptr), shape=matrix.shape)
    return PreparedMannWhitney(ranks, zero_ranks, tie_terms, workers)


@dataclass(frozen=True)
class MannWhitney:
    kind = "p"
    correction: BY = field(default_factory=BY)
    correlation_threshold: float | None = None

    def __post_init__(self):
        if self.correlation_threshold is not None and not 0 < self.correlation_threshold < 1:
            raise ValueError("Invalid Pearson threshold")

    def test(self, matrix: sparse.spmatrix, y: np.ndarray) -> dict[str, np.ndarray]:
        """Test sample×feature values against binary labels; return aligned p-values/effects."""
        p, effects = mann_whitney_many(matrix, [np.asarray(y)])
        return dict(p=p[0], effect=effects)

    def batch_test(self, matrix: sparse.spmatrix, targets: np.ndarray,
                   prepared: PreparedMannWhitney | None = None) -> np.ndarray:
        """Map shuffle×sample labels to shuffle×feature p-values, reusing pooled ranks."""
        if prepared is not None:
            return prepared.test(targets)
        return np.asarray(mann_whitney_many(matrix, targets)[0])

    def family_size(self, candidates: int, original: int) -> int:
        """MW/BY tests the complete eligible represented-feature family."""
        return candidates

    def prepare(self, matrix: sparse.spmatrix) -> PreparedMannWhitney | None:
        """Cache label-independent ranks once; other dtypes use the reference path."""
        return prepare_mann_whitney(matrix) if matrix.dtype == np.float64 else None

    def test_settings(self):
        return {k: v for k, v in asdict(self).items() if k not in {"correction", "correlation_threshold"}}

    def dependencies(self):
        return (type(self).test, type(self).batch_test, type(self).test_settings, type(self).prepare,
                mann_whitney_many, prepare_mann_whitney, PreparedMannWhitney)

    def selection_dependencies(self):
        return (type(self).select, type(self).family_size, type(self.correction).adjust, greedy_filter)

    def select(self, evidence: dict, matrix: sparse.spmatrix, ids: np.ndarray,
                family_size: int, rank_limit: int) -> dict[str, np.ndarray]:
        """Return unscreened order, BY discoveries and q-values in matrix-column coordinates.

        IDs break ranking ties. Optional filtering compares each candidate to
        every accepted column; masked-p BY retains the complete tested family.
        """
        p = evidence["p"]
        order = np.lexsort((ids, p))
        q = self.correction.adjust(p)
        native = order[q[order] <= self.correction.alpha]
        if self.correlation_threshold is not None:
            accepted = greedy_filter(matrix, native, self.correlation_threshold)
            working = np.ones(len(p))
            working[accepted] = p[accepted]
            q = self.correction.adjust(working)
            native = accepted[q[accepted] <= self.correction.alpha]
            order = greedy_filter(matrix, order, self.correlation_threshold, rank_limit)
        return dict(order=order[:rank_limit], native=native, q=q)

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
