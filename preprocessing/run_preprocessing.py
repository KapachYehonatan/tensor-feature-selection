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
TARGET_COLUMN = "disease"
NEGATIVE_LABEL = "healthy"
POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL", "CRC")
PREVALENCE_FRACTION = 0.05
MINIMUM_SAMPLES = 5
OUTPUT = PROJECT / "outputs/preprocessing" / DATASET

sys.path.insert(0, str(PROJECT))
from research_utils import prevalence_mask, write_lines  # noqa: E402


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
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if metadata[sample_column].duplicated().any():
        raise ValueError("Metadata sample IDs must be unique")
    metadata = metadata.set_index(sample_column).reindex(sample_ids)
    labels = metadata[TARGET_COLUMN].astype(str)
    keep = labels.isin([NEGATIVE_LABEL, POSITIVE_LABEL]).to_numpy()
    if not keep.any():
        raise ValueError("No samples match the configured binary labels")
    selected = metadata.iloc[np.flatnonzero(keep)].copy()
    selected.insert(0, "sample_id", selected.index.astype(str))
    selected["y"] = labels[keep].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).to_numpy()
    return selected.reset_index(drop=True), keep


if not 0 < PREVALENCE_FRACTION <= 1:
    raise ValueError("PREVALENCE_FRACTION must be in (0, 1]")

study = SOURCE_ROOT / DATASET
matrix_path = study / f"{DATASET}_gene_families.mtx"
feature_path = study / f"{DATASET}_genes.txt"
sample_path = study / f"{DATASET}_samples.txt"
metadata_path = study / f"{DATASET}_metadata.tsv"
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
matrix = matrix[labeled_rows][:, valid].astype(np.float32)
matrix.eliminate_zeros()
if matrix.data.size and matrix.data.min() < 0:
    raise ValueError("Expected nonnegative abundances")

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

# Step 3: report the whole-data filter only as an audit preview. Evaluation
# recomputes this mask from each training fold to prevent leakage.
mask, support, threshold = prevalence_mask(
    matrix,
    PREVALENCE_FRACTION,
    MINIMUM_SAMPLES,
)
total_mass = float(matrix.sum())
retained_mass = float(matrix[:, mask].sum())

# Step 4: save the one necessary sparse cache and compact mappings.
OUTPUT.mkdir(parents=True, exist_ok=True)
sparse.save_npz(OUTPUT / "X_valid.npz", matrix, compressed=True)
np.savez_compressed(
    OUTPUT / "edges.npz",
    pair_gene=pair_gene,
    pair_bacteria=pair_bacteria,
    global_support=support,
    global_mask=mask,
)
samples[["sample_id", TARGET_COLUMN, "y"]].to_csv(
    OUTPUT / "samples.csv", index=False
)
write_lines(OUTPUT / "genes.txt.gz", genes)
write_lines(OUTPUT / "bacteria.txt.gz", bacteria)
pd.DataFrame(
    [{
        "dataset": DATASET,
        "samples": matrix.shape[0],
        "negative": int((samples["y"] == 0).sum()),
        "positive": int((samples["y"] == 1).sum()),
        "valid_pairs": matrix.shape[1],
        "retained_pairs_preview": int(mask.sum()),
        "minimum_sample_support": threshold,
        "retained_mass_fraction": retained_mass / total_mass,
        "genes": len(genes),
        "bacteria": len(bacteria),
        "source_matrix": str(matrix_path),
    }]
).to_csv(OUTPUT / "filter_summary.csv", index=False, float_format="%.3f")

print(
    f"Samples: {matrix.shape[0]}; negative: {(samples['y'] == 0).sum()}; "
    f"positive: {(samples['y'] == 1).sum()}"
)
print(
    f"Valid pairs: {matrix.shape[1]:,}; preview retained: {mask.sum():,}; "
    f"sample threshold: {threshold}; retained mass: {retained_mass / total_mass:.3f}"
)
print(f"Saved prepared data to {OUTPUT}")
