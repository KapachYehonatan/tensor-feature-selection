"""Explicit cMD input export; source files are read-only and values are unchanged."""
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from execution import cache
from datasets.loading import unique_day_zero_stool_mask


def pair_dictionary(path):
    valid, genes, bacteria, pg, pb = [], {}, {}, [], []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            name = line.rstrip("\n")
            if "|" not in name:
                continue
            gene, bacterium = name.split("|", 1)
            if not gene or not bacterium or bacterium == "unclassified":
                continue
            if gene not in genes:
                genes[gene] = len(genes)
            if bacterium not in bacteria:
                bacteria[bacterium] = len(bacteria)
            valid.append(index)
            pg.append(genes[gene])
            pb.append(bacteria[bacterium])
    return np.asarray(valid, int), np.asarray(pb, np.int32), np.asarray(pg, np.int32), genes, bacteria


def prepare(name, source_directory, source_study, *, target_column, negative_label,
            positive_label, independent_day_zero=False, destination=None):
    """Export a configured binary cohort; no disease, path, or cohort inference."""
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError("Invalid dataset name")
    directory = Path(source_directory)
    paths = {kind: directory / f"{source_study}_{suffix}" for kind, suffix in
             dict(matrix="gene_families.mtx", features="genes.txt", samples="samples.txt", metadata="metadata.tsv").items()}
    sample_ids = [line.strip() for line in paths["samples"].read_text().splitlines() if line.strip()]
    metadata = pd.read_csv(paths["metadata"], sep="\t", low_memory=False)
    sample_column = "sample_id" if "sample_id" in metadata else "SampleID"
    if metadata[sample_column].duplicated().any():
        raise ValueError("Duplicate source sample IDs")
    metadata = metadata.set_index(sample_column).reindex(sample_ids)
    labels = metadata[target_column].astype(str)
    keep = labels.isin([negative_label, positive_label]).to_numpy(copy=True)
    if independent_day_zero:
        keep &= unique_day_zero_stool_mask(metadata.subject_id.to_numpy(), metadata.body_site.to_numpy(),
            pd.to_numeric(metadata.days_from_first_collection, errors="coerce").to_numpy(),
            pd.to_numeric(metadata.number_reads, errors="coerce").to_numpy(), sample_ids)
    samples = metadata.iloc[np.flatnonzero(keep)].copy()
    samples.insert(0, "sample_id", samples.index.astype(str))
    samples["y"] = labels[keep].map({negative_label: 0, positive_label: 1}).to_numpy()
    samples = samples.reset_index(drop=True)
    if samples.empty or set(samples.y) != {0, 1} or ("subject_id" in samples and samples.subject_id.duplicated().any()):
        raise ValueError("Cohort must contain independent participants and both labels")
    valid, pb, pg, genes, bacteria = pair_dictionary(paths["features"])
    raw = mmread(paths["matrix"], spmatrix=True)
    if raw.shape[1] == len(sample_ids):
        matrix = sparse.csr_matrix(raw.T)
    elif raw.shape[0] == len(sample_ids):
        matrix = sparse.csr_matrix(raw)
    else:
        raise ValueError("Neither matrix dimension matches source samples")
    matrix = matrix[keep][:, valid]
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Supplied values must be finite and nonnegative")
    observed = np.asarray(matrix.getnnz(axis=0)).ravel() > 0
    matrix, pb, pg = matrix[:, observed], pb[observed], pg[observed]
    if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
        raise ValueError("An included sample has zero classified abundance")
    output = cache.CACHE / "inputs" / name if destination is None else Path(destination)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing input export")
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "X_valid.npz", matrix)
    np.savez_compressed(output / "edges.npz", pair_bacteria=pb, pair_gene=pg)
    columns = ["sample_id", "y"] + (["subject_id"] if "subject_id" in samples else [])
    samples[columns].to_csv(output / "samples.csv", index=False)
    for filename, values in (("genes.txt.gz", genes), ("bacteria.txt.gz", bacteria)):
        with gzip.open(output / filename, "wt") as handle:
            handle.writelines(value + "\n" for value in values)
    pd.DataFrame([dict(dataset=name, source_study=source_study, samples=len(samples), pairs=matrix.shape[1],
                       negative=int((samples.y == 0).sum()), positive=int((samples.y == 1).sum()),
                       genes=len(genes), bacteria=len(bacteria), independent_day_zero=independent_day_zero)]).to_csv(output / "filter_summary.csv", index=False)
    return output
