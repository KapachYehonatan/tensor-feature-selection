"""Unscreened MW ranking and separate complete-family discoveries."""
import numpy as np
from scipy import sparse, stats
from dataclasses import asdict, dataclass, field
from methods.correction import BY
from methods.selectors.correlation import greedy_filter

MANN_WHITNEY_CHUNK = 2000


@dataclass(frozen=True)
class MannWhitney:
    kind = "p"
    correction: BY = field(default_factory=BY)
    correlation_threshold: float | None = None

    def __post_init__(self):
        if self.correlation_threshold is not None and not 0 < self.correlation_threshold < 1:
            raise ValueError("Invalid Pearson threshold")

    def test(self, matrix, y):
        p, effects = mann_whitney_many(matrix, [np.asarray(y)])
        return dict(p=p[0], effect=effects)

    def batch_test(self, matrix, targets, prepared=None):
        return np.asarray(mann_whitney_many(matrix, targets)[0])

    def family_size(self, candidates, original):
        return candidates

    def prepare(self, matrix):
        return None

    def test_settings(self):
        return {k: v for k, v in asdict(self).items() if k not in {"correction", "correlation_threshold"}}

    def dependencies(self):
        return (type(self).test, type(self).batch_test, type(self).test_settings, mann_whitney_many)

    def selection_dependencies(self):
        return (type(self).select, type(self).family_size, type(self.correction).adjust, greedy_filter)

    def select(self, evidence, matrix, ids, family_size, rank_limit):
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
