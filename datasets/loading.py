"""Read unfiltered supplied-value inputs without changing cohort membership."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from execution import cache


@dataclass
class Dataset:
    name: str
    path: Path
    values: sparse.csr_matrix
    samples: pd.DataFrame
    pair_bacteria: np.ndarray
    pair_gene: np.ndarray
    fingerprint: dict


def validate(values, samples, pair_bacteria, pair_gene):
    if (values.shape != (len(samples), len(pair_bacteria)) or len(pair_gene) != len(pair_bacteria)
            or not np.isfinite(values.data).all() or np.any(values.data < 0)
            or np.any(np.asarray(values.sum(axis=1)).ravel() <= 0)
            or samples.sample_id.isna().any() or samples.sample_id.duplicated().any()
            or ("subject_id" in samples and (samples.subject_id.isna().any() or samples.subject_id.duplicated().any()))
            or set(samples.y.unique()) != {0, 1} or not np.issubdtype(pair_bacteria.dtype, np.integer)
            or not np.issubdtype(pair_gene.dtype, np.integer) or np.any(pair_bacteria < 0) or np.any(pair_gene < 0)):
        raise ValueError("Invalid sparse values, mappings, labels, or independent participants")


def load(name, root=None):
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError("Dataset name must be one plain directory name")
    directory = (cache.CACHE / "inputs" if root is None else Path(root)) / name
    samples = pd.read_csv(directory / "samples.csv", dtype={"sample_id": str})
    values = sparse.load_npz(directory / "X_valid.npz").tocsr()
    values.sum_duplicates()
    values.eliminate_zeros()
    with np.load(directory / "edges.npz") as saved:
        pb, pg = saved["pair_bacteria"], saved["pair_gene"]
    validate(values, samples, pb, pg)
    inputs = {filename: cache.file_hash(directory / filename)
              for filename in ("X_valid.npz", "samples.csv", "edges.npz")}
    return Dataset(name, directory, values, samples, pb, pg, inputs)


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
