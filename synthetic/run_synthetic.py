#!/usr/bin/env python3
"""Create small sparse datasets with known bacteria, gene, and pair signals."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT = Path("/home/kapachy/Microbium/tensor_feature_selection")
OUTPUT_ROOT = PROJECT / "outputs/preprocessing"
SIGNAL_MODES = ["bacteria", "gene", "pair"]
N_SAMPLES = 120
N_BACTERIA = 8
N_GENES = 40
ZERO_PROBABILITY = 0.85
SIGNAL_SIZE = 0.25
SEED = 42

sys.path.insert(0, str(PROJECT))
from research_utils import sample_support, write_lines  # noqa: E402


def generate(mode, seed):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], N_SAMPLES // 2)
    rng.shuffle(y)
    tensor = rng.lognormal(-3.0, 0.7, size=(N_SAMPLES, N_BACTERIA, N_GENES))
    tensor[rng.random(tensor.shape) < ZERO_PROBABILITY] = 0

    if mode == "bacteria":
        # Move mass between two bacteria while preserving every gene margin.
        tensor[:, :2, :] = 1.0
        tensor[y == 1, 0, :] += SIGNAL_SIZE
        tensor[y == 1, 1, :] -= SIGNAL_SIZE
        truth = ["bacterium_0"]
    elif mode == "gene":
        # Move mass between two genes while preserving every bacteria margin.
        tensor[:, :, :2] = 1.0
        tensor[y == 1, :, 0] += SIGNAL_SIZE
        tensor[y == 1, :, 1] -= SIGNAL_SIZE
        truth = ["gene_0"]
    elif mode == "pair":
        # A 2x2 cycle changes pair dependence but preserves both margins.
        tensor[:, :2, :2] = 1.0
        positive = y == 1
        tensor[positive, 0, 0] += SIGNAL_SIZE
        tensor[positive, 1, 1] += SIGNAL_SIZE
        tensor[positive, 0, 1] -= SIGNAL_SIZE
        tensor[positive, 1, 0] -= SIGNAL_SIZE
        truth = [
            "gene_0|bacterium_0", "gene_1|bacterium_1",
            "gene_1|bacterium_0", "gene_0|bacterium_1",
        ]
    else:
        raise ValueError(f"Unknown signal mode: {mode}")

    tensor *= rng.lognormal(0, 0.2, size=(N_SAMPLES, 1, 1))
    pair_gene = np.tile(np.arange(N_GENES), N_BACTERIA)
    pair_bacteria = np.repeat(np.arange(N_BACTERIA), N_GENES)
    matrix = sparse.csr_matrix(tensor.reshape(N_SAMPLES, -1).astype(np.float32))
    return matrix, pair_gene, pair_bacteria, y, truth


for number, mode in enumerate(SIGNAL_MODES):
    matrix, pair_gene, pair_bacteria, y, truth = generate(mode, SEED + number)
    output = OUTPUT_ROOT / f"synthetic_{mode}"
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "X_valid.npz", matrix, compressed=True)
    support = sample_support(matrix)
    np.savez_compressed(
        output / "edges.npz",
        pair_gene=pair_gene.astype(np.int32),
        pair_bacteria=pair_bacteria.astype(np.int32),
        global_support=support,
        global_mask=support >= max(10, int(np.ceil(0.05 * N_SAMPLES))),
    )
    pd.DataFrame({
        "sample_id": [f"sample_{i}" for i in range(N_SAMPLES)],
        "disease": np.where(y == 1, mode, "healthy"),
        "y": y,
    }).to_csv(output / "samples.csv", index=False)
    write_lines(output / "genes.txt.gz", [f"gene_{i}" for i in range(N_GENES)])
    write_lines(output / "bacteria.txt.gz", [f"bacterium_{i}" for i in range(N_BACTERIA)])
    pd.DataFrame({"level": mode, "feature": truth}).to_csv(
        output / "true_signals.csv", index=False
    )
    print(
        f"synthetic_{mode}: samples={N_SAMPLES}; pairs={matrix.shape[1]}; "
        f"nnz={matrix.nnz}; truth={len(truth)}"
    )
