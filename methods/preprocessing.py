"""Input validation and training-local filtering of supplied values."""
import numpy as np
from scipy import sparse

def unique_day_zero_stool_mask(subjects, body_sites, days, reads, sample_ids):
    """Select one day-zero stool sample per subject, preferring read depth."""
    subjects = np.asarray(subjects, dtype=object)
    body_sites = np.asarray(body_sites, dtype=object)
    days = np.asarray(days, dtype=float)
    reads = np.asarray(reads, dtype=float)
    sample_ids = np.asarray(sample_ids, dtype=str)
    if len({len(subjects), len(body_sites), len(days), len(reads), len(sample_ids)}) != 1:
        raise ValueError("Baseline-selection columns must have equal lengths")

    selected = np.zeros(len(subjects), dtype=bool)
    best = {}
    for index in np.flatnonzero(
        np.array([isinstance(value, str) and bool(value) for value in subjects])
        & (np.char.lower(body_sites.astype(str)) == "stool")
        & (days == 0)
    ):
        subject = subjects[index]
        incumbent = best.get(subject)
        read_count = reads[index] if np.isfinite(reads[index]) else -np.inf
        if incumbent is None:
            best[subject] = index
            continue
        incumbent_reads = reads[incumbent] if np.isfinite(reads[incumbent]) else -np.inf
        if read_count > incumbent_reads or (
            read_count == incumbent_reads and sample_ids[index] < sample_ids[incumbent]
        ):
            best[subject] = index
    selected[list(best.values())] = True
    return selected


def sample_support(matrix):
    """Count the samples with a positive value for each feature."""
    matrix = sparse.csr_matrix(matrix)
    detected = matrix.copy()
    detected.data = detected.data > 0
    detected.eliminate_zeros()
    return np.asarray(detected.getnnz(axis=0)).ravel().astype(np.int32)


def prevalence_mask(matrix, fraction=0.05, minimum_samples=5):
    if not 0 < fraction <= 1:
        raise ValueError("Prevalence fraction must be in (0, 1]")
    threshold = max(minimum_samples, int(np.ceil(fraction * matrix.shape[0])))
    support = sample_support(matrix)
    return support >= threshold, support, threshold


def retained_pair_values(matrix, columns):
    """Select supplied values unchanged; reject invalid or empty retained samples."""
    values = sparse.csr_matrix(matrix[:, columns], dtype=np.float64)
    if not np.isfinite(values.data).all() or np.any(values.data < 0):
        raise ValueError("Feature values must be finite and nonnegative")
    if values.shape[1] and np.any(np.asarray(values.sum(axis=1)).ravel() <= 0):
        raise ValueError("Samples have no retained abundance")
    return values



def validate_samples(matrix, samples, pb, pg):
    if (matrix.shape != (len(samples), len(pb)) or len(pg) != len(pb) or
            not np.isfinite(matrix.data).all() or np.any(matrix.data < 0) or
            np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0) or
            samples.sample_id.isna().any() or samples.sample_id.duplicated().any() or
            ('subject_id' in samples and (samples.subject_id.isna().any() or samples.subject_id.duplicated().any())) or
            set(samples.y.unique()) != {0, 1} or not np.issubdtype(pb.dtype, np.integer) or
            not np.issubdtype(pg.dtype, np.integer) or np.any(pb < 0) or np.any(pg < 0)):
        raise ValueError('Invalid independent samples, labels, abundance matrix or pair identifiers')

