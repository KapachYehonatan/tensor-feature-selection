#!/usr/bin/env python3
"""Prepare one cMD study without modifying the source files."""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


PROJECT = Path("/home/kapachy/Microbium/tensor_feature_selection")
SOURCE_ROOT = Path("/home/kapachy/Microbium/projects/cMD_downloads")
DATASET = os.environ.get("DATASET", "WirbelJ_2018")
SOURCE_DATASETS = {
    "HMP_2019_ibdmdb_unique": "HMP_2019_ibdmdb",
    "NielsenHB_2014_unique": "NielsenHB_2014",
}
SOURCE_DATASET = SOURCE_DATASETS.get(DATASET, DATASET)
TARGET_COLUMN = "disease"
NEGATIVE_LABEL = "healthy"
POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL", "CRC")
OUTPUT = PROJECT / "outputs/.cache/inputs" / DATASET

sys.path.insert(0, str(PROJECT))
from methods.preprocessing import (  # noqa: E402
    unique_day_zero_stool_mask,
)

from experiments.cache import write_lines


def parse_pair_names(path):
    """Keep classified stratified rows and encode their two biological IDs."""
    valid_indices = []
    gene_lookup, bacteria_lookup = {}, {}
    genes, bacteria = [], []
    pair_gene, pair_bacteria = [], []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            name = line.rstrip("\n")
            if "|" not in name:
                continue
            gene, bacterium = name.split("|", 1)
            if not gene or not bacterium or bacterium == "unclassified":
                continue
            if gene not in gene_lookup:
                gene_lookup[gene] = len(genes)
                genes.append(gene)
            if bacterium not in bacteria_lookup:
                bacteria_lookup[bacterium] = len(bacteria)
                bacteria.append(bacterium)
            valid_indices.append(index)
            pair_gene.append(gene_lookup[gene])
            pair_bacteria.append(bacteria_lookup[bacterium])
    return (
        np.asarray(valid_indices, dtype=np.int64),
        np.asarray(pair_gene, dtype=np.int32),
        np.asarray(pair_bacteria, dtype=np.int32),
        genes,
        bacteria,
    )


def align_metadata(path, sample_ids):
    metadata = pd.read_csv(path, sep="\t", low_memory=False)
    sample_column = "sample_id" if "sample_id" in metadata else "SampleID"
    required = {sample_column, TARGET_COLUMN}
    if DATASET in SOURCE_DATASETS:
        required.update({"subject_id", "body_site", "days_from_first_collection", "number_reads"})
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if metadata[sample_column].duplicated().any():
        raise ValueError("Metadata sample IDs must be unique")
    metadata = metadata.set_index(sample_column).reindex(sample_ids)
    labels = metadata[TARGET_COLUMN].astype(str)
    keep = labels.isin([NEGATIVE_LABEL, POSITIVE_LABEL]).to_numpy(copy=True)
    if DATASET in SOURCE_DATASETS:
        keep &= unique_day_zero_stool_mask(
            metadata["subject_id"].to_numpy(),
            metadata["body_site"].to_numpy(),
            pd.to_numeric(metadata["days_from_first_collection"], errors="coerce"),
            pd.to_numeric(metadata["number_reads"], errors="coerce"),
            sample_ids,
        )
    if not keep.any():
        raise ValueError("No samples match the configured binary labels")
    selected = metadata.iloc[np.flatnonzero(keep)].copy()
    if DATASET in SOURCE_DATASETS and selected["subject_id"].duplicated().any():
        raise RuntimeError("Unique baseline selection retained a repeated subject")
    selected.insert(0, "sample_id", selected.index.astype(str))
    selected["y"] = labels[keep].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).to_numpy()
    return selected.reset_index(drop=True), keep


study = SOURCE_ROOT / SOURCE_DATASET
matrix_path = study / f"{SOURCE_DATASET}_gene_families.mtx"
feature_path = study / f"{SOURCE_DATASET}_genes.txt"
sample_path = study / f"{SOURCE_DATASET}_samples.txt"
metadata_path = study / f"{SOURCE_DATASET}_metadata.tsv"
for path in [matrix_path, feature_path, sample_path, metadata_path]:
    if not path.exists():
        raise FileNotFoundError(path)

# Step 1: parse the biological pair dictionary without loading feature values.
sample_ids = [line.strip() for line in sample_path.read_text().splitlines() if line.strip()]
valid, pair_gene, pair_bacteria, genes, bacteria = parse_pair_names(feature_path)
samples, labeled_rows = align_metadata(metadata_path, sample_ids)

# Step 2: load the sparse coordinate matrix and standardize samples as rows.
# This can take hours for the largest studies; run one study deliberately.
raw = mmread(matrix_path, spmatrix=True)
if raw.shape[1] == len(sample_ids):
    matrix = sparse.csr_matrix(raw.T)
elif raw.shape[0] == len(sample_ids):
    matrix = sparse.csr_matrix(raw)
else:
    raise ValueError(f"Neither matrix dimension matches {len(sample_ids)} samples")
del raw
matrix = matrix[labeled_rows][:, valid]
matrix.eliminate_zeros()
if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
    raise ValueError("Expected finite nonnegative feature values")

# Remove pair columns absent from every labeled sample.
observed = np.asarray(matrix.getnnz(axis=0)).ravel() > 0
matrix = matrix[:, observed]
pair_gene, pair_bacteria = pair_gene[observed], pair_bacteria[observed]
empty = np.flatnonzero(matrix.getnnz(axis=1) == 0)
if len(empty):
    raise ValueError(
        f"{len(empty)} samples have no classified pair abundances: "
        f"{samples.iloc[empty]['sample_id'].tolist()[:10]}"
    )

# Step 3: save supplied values and mappings; prevalence filtering is training-local.
OUTPUT.mkdir(parents=True, exist_ok=True)
sparse.save_npz(OUTPUT / "X_valid.npz", matrix, compressed=True)
np.savez_compressed(
    OUTPUT / "edges.npz",
    pair_gene=pair_gene,
    pair_bacteria=pair_bacteria,
)
sample_columns = ["sample_id", TARGET_COLUMN, "y"]
if DATASET in SOURCE_DATASETS:
    sample_columns[1:1] = [
        "subject_id", "body_site", "days_from_first_collection", "number_reads"
    ]
samples[sample_columns].to_csv(
    OUTPUT / "samples.csv", index=False
)
write_lines(OUTPUT / "genes.txt.gz", genes)
write_lines(OUTPUT / "bacteria.txt.gz", bacteria)
pd.DataFrame(
    [{
        "dataset": DATASET,
        "source_dataset": SOURCE_DATASET,
        "cohort_rule": "one_day_zero_stool_per_subject" if DATASET in SOURCE_DATASETS else "all_labeled_samples",
        "samples": matrix.shape[0],
        "subjects": samples["subject_id"].nunique() if "subject_id" in samples else matrix.shape[0],
        "negative": int((samples["y"] == 0).sum()),
        "positive": int((samples["y"] == 1).sum()),
        "valid_pairs": matrix.shape[1],
        "genes": len(genes),
        "bacteria": len(bacteria),
        "source_matrix": str(matrix_path),
    }]
).to_csv(OUTPUT / "filter_summary.csv", index=False, float_format="%.3f")

print(
    f"Samples: {matrix.shape[0]}; negative: {(samples['y'] == 0).sum()}; "
    f"positive: {(samples['y'] == 1).sum()}"
)
print(f"Valid pairs: {matrix.shape[1]:,}; genes: {len(genes):,}; bacteria: {len(bacteria):,}")
print(f"Saved prepared data to {OUTPUT}")
