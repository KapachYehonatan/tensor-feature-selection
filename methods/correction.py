"""Corrections act on complete tested families, never prediction Top-K subsets."""
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BY:
    alpha: float = .10

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("Invalid BY level")

    def adjust(self, p):
        p = np.asarray(p, dtype=float)
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid complete p-value family")
        return stats.false_discovery_control(p, method="by") if len(p) else p.copy()


@dataclass(frozen=True)
class EBH:
    alpha: float = .10

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("Invalid e-BH level")

    def select(self, log_e, ids, family_size):
        log_e, ids = np.asarray(log_e), np.asarray(ids)
        if (not np.isscalar(family_size) or not np.isfinite(family_size) or int(family_size) != family_size
                or family_size < len(ids) or family_size < 0 or log_e.ndim != 1 or ids.shape != log_e.shape
                or (len(ids) and not np.issubdtype(ids.dtype, np.integer))
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= family_size)
                or np.isnan(log_e).any() or np.isposinf(log_e).any()):
            raise ValueError("Invalid full-family evidence")
        order = np.lexsort((ids, -log_e))
        if not len(order):
            return order
        passing = np.flatnonzero(log_e[order] + np.log(np.arange(1, len(order) + 1)) >= np.log(family_size / self.alpha))
        return order[:int(passing[-1] + 1)] if len(passing) else order[:0]
