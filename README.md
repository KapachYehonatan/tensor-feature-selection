# Robust feature selection in ultra-high-dimensional microbiome data

This project studies feature selection and statistical multiple testing for sparse
microbiome gene abundances with a **(bacterium, gene)** structure. The current
comparison includes bacteria-collapsed, gene-collapsed, Bernoulli-join mean, SHOE,
MW → e-BH, Hurdle-product → e-BH, and Hurdle-average → e-BH. These are the current
methods, not a permanent restriction on the research.

## Code

- `methods/preprocessing.py` and `representations.py`: input validation,
  training-local filtering, raw pairs, and group means.
- `methods/graphs/`: adaptive Bernoulli graph preparation and exact join/Leiden worker.
- `methods/selectors/`: MW/BH/BY, SHOE, and MW/Hurdle e-value selectors.
- `metrics/`: prediction metrics and complete-set pairwise absolute correlation.
- `shuffled_fdp_test/`: shuffled discovery counts and the count-comparison p-value.
- `experiments/`: CV/full-dataset scientific flows and strict cache validation.
- `scripts/`: input export, sequential execution, resource launcher, and plotting.
- `tests/test_experiment.py`: small scientific and cache-integrity fixtures.
- `docs/`: method theory notes.

## Scientific protocol

Keep finite nonnegative sparse values as supplied: no sample normalization, log,
or square-root transformation. Learn prevalence support ≥ max(5,ceil(.05*n)) and
constant exclusion locally. Representations average retained pair values with
fixed training-defined denominators; raw-pair selectors do not aggregate.

Prediction uses identical seed-42 five-fold splits, K=1–10 and {10,20,50,75},
without tuning. Use training-fitted `StandardScaler(with_mean=False)` and balanced
L2 logistic regression C=1. Keep mean-fold/SD and pooled OOF metrics distinct.
Top-K prediction rankings are not FDR-controlled discovery sets.

Full-dataset findings use all samples separately from CV. Baselines define BH/BY
discoveries at alpha=.10; the four e-value methods use native e-BH. SHOE alone
uses an honest internal 50/50 score/evidence split. Average pairwise absolute
correlation uses complete discovery sets: undefined below two features and
skipped above 10,000, without sampling. The Shuffled FDP test compares real
discoveries with 100 paired class-preserving label shuffles; its p-value requires
the binomial/normal approximation and is not an FDR certificate. Findings remain
exploratory. No separate shuffled-test diagnostics or report are generated.

Graphs start at absolute phi .70 and adapt upward by .05 only for resource
guards. Keep the 500-million-edge limit and ordinary weighted-modularity Leiden
convergence, resolution 1 and seed 17729. Source data under
`/home/kapachy/Microbium/projects/cMD_downloads` and the sibling Bernoulli-join
repository/environment remain untouched.

## Usage and outputs

Regenerate the eight public figures, without scientific fitting:

```bash
.venv/bin/python scripts/plot_comparison.py
```

`outputs/` contains four `AUC_average_k_…` / `AUC_overview_k_…` figures and the
pairwise-correlation, −log10(p-value), and real-discovery overviews. Supporting
inputs/results are private under `outputs/.cache/`: `inputs`, `cv`,
`full_dataset`, `shuffled_fdp_test`, `plot_data`, and `provenance`.
The additional `average_pairwise_absolute_correlation_k_10_20_50_75.png` shows
full-dataset correlations of unscreened score-ranked Top-K features for all seven
methods, not significance-filtered subsets or CV-fold averages.

For a scientific run/resumption, use `scripts/launch_experiment.py` under `nohup`
inside a persistent user service with `MemoryMax=200G` and `MemorySwapMax=0`.
The supervisor writes logs/PIDs under `logs/`; use one numerical thread, four
join workers, and one Leiden worker. HMP and QinJ run first. Completed compatible
caches resume without scientific refitting; incompatible inputs/settings/code
fail rather than overwriting results. Plotting, tests, and documentation are not
part of scientific fingerprints.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
```

Original protocols and the exact cleanup/compatibility records are retained
privately in `outputs/.cache/provenance/`. Superseded runs are permanently
deleted; the reorganization itself does not rerun any real experiment.
