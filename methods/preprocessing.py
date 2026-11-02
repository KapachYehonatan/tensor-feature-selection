"""Input validation and training-local filtering of supplied values."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Preprocessing:
    prevalence: float = .05
    minimum_samples: int = 5

    def __post_init__(self):
        if not 0 < self.prevalence <= 1 or self.minimum_samples < 1:
            raise ValueError("Invalid prevalence settings")

    def fit(self, matrix: sparse.spmatrix, training: np.ndarray) -> tuple[np.ndarray, int]:
        """Learn retained pair-column IDs and minimum support from training rows."""
        mask, _, cutoff = prevalence_mask(matrix[training], self.prevalence, self.minimum_samples)
        return np.flatnonzero(mask), cutoff

def sample_support(matrix: sparse.spmatrix) -> np.ndarray:
    """Count the samples with a positive value for each feature."""
    matrix = sparse.csr_matrix(matrix)
    detected = matrix.copy()
    detected.data = detected.data > 0
    detected.eliminate_zeros()
    return np.asarray(detected.getnnz(axis=0)).ravel().astype(np.int32)


def prevalence_mask(matrix: sparse.spmatrix, fraction: float = .05,
                    minimum_samples: int = 5) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a pair-column mask, observed supports, and the required support."""
    if not 0 < fraction <= 1:
        raise ValueError("Prevalence fraction must be in (0, 1]")
    threshold = max(minimum_samples, int(np.ceil(fraction * matrix.shape[0])))
    support = sample_support(matrix)
    return support >= threshold, support, threshold


def retained_pair_values(matrix: sparse.spmatrix, columns: np.ndarray) -> sparse.csr_matrix:
    """Select supplied values unchanged; reject invalid or empty retained samples."""
    values = sparse.csr_matrix(matrix[:, columns], dtype=np.float64)
    if not np.isfinite(values.data).all() or np.any(values.data < 0):
        raise ValueError("Feature values must be finite and nonnegative")
    if values.shape[1] and np.any(np.asarray(values.sum(axis=1)).ravel() <= 0):
        raise ValueError("Samples have no retained abundance")
    return values

