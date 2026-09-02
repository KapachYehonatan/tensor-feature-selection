# Data provenance

Source datasets are read directly and never modified:

`/home/kapachy/Microbium/projects/cMD_downloads`

Each study contains a Matrix Market coordinate matrix, ordered feature and
sample text files, and clinical metadata. Many studies also contain previously
cleaned SciPy sparse matrices and earlier experiment outputs. The primary
prevalence analysis starts from the raw Matrix Market data because the existing
cleaned files appear to use a stricter 15% prevalence threshold.

The preprocessing script records the selected source files, dimensions, target
mapping, and filter settings in its output summary. No source file is copied
into this directory.

## Synthetic tensor format

`scripts/generate_synthetic.py` writes each generated dataset below
`outputs/synthetic/<preset>/n_<samples>/seed_<seed>/<scenario>/`. `tensor.npz`
contains five arrays: `sample`, `bacterium`, `gene`, `value`, and the
three-dimensional `shape`. Every row describes one positive tensor entry;
missing entries are zero. Values are float32 and indices use the smallest safe
unsigned integer type.

`target.npy` is present only for target scenarios. Compressed CSV files record
true bacteria clusters, true gene clusters, block prevalences, and planted
logistic coefficients. `target_model.npz` stores the true linear predictor and
Bernoulli probability. `metadata.json` records the seed, named random streams,
realized sparsity, target prevalence, configuration hash, and complete preset.
Target scenarios hard-link the shared noisy `tensor.npz`, avoiding duplicate
storage while preserving the same per-scenario paths. The dense tensor is never
stored or loaded.
