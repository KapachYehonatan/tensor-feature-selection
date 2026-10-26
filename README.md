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
discoveries with 999 paired class-preserving label shuffles, restricted to
Bacteria BY, Gene BY, Join BY and Bins e-value (historical cache ID
`exact_categorical`). Other methods remain in the prediction/correlation experiments.
H0 is label
exchangeability conditional on the feature matrix and class counts; the targeted
H1 shifts the discovery count upward. The p-value is `(1 + shuffles >= real)/(999 + 1)`,
including ties. It requires exchangeability, not independent features, and is not
an FDR certificate. Findings remain exploratory; no separate diagnostics are generated.

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

All eight public figures compare Bacteria BY, Gene BY, Join BY and Bins e-value.
BY labels identify discovery rules; prediction and score-ranked correlation
curves still use unscreened Top-K features, not BY-filtered subsets.
`outputs/` contains four `AUC_average_k_…` / `AUC_overview_k_…` figures and the
pairwise-correlation, −log10(p-value), and real-discovery overviews. Supporting
inputs/results are private under `outputs/.cache/`: `inputs`, `cv`,
`full_dataset`, `shuffled_fdp_test`, `plot_data`, and `provenance`.
The additional `average_pairwise_absolute_correlation_k_10_20_50_75.png` shows
full-dataset correlations of unscreened score-ranked Top-K features for these four
methods, not significance-filtered subsets or CV-fold averages.

Bins e-value uses one three-bin log-likelihood score at fixed lambda=1,
`E=exp(T)/E0(exp(T))`, with exact conditional normalization and e-BH at .10.
There is no presence/absence alternative, temperature search, maximum or boost.
`scripts/search_evalue_power.py` fits it to cached full-data raw pairs; private
results live under `outputs/.cache/evalue_power/bins/`.
`scripts/evaluate_bins_cv.py` adds Bins e-value predictions from existing fold-local
raw-pair caches under `outputs/.cache/cv/bins/`.
It fits bins/evidence only on each training fold and never constructs graphs.

Run `scripts/run_shuffled_test.py` with the resource supervisor for one shuffle
test per method/dataset. The sole budget is `SHUFFLE_COUNT=999` in
`experiments/cache.py`. Ordinary batch checkpoints resume at any completed count;
there is no initial subset, separate extension, or prefix replay. Full-data
selection evaluates real labels only; all label shuffles belong to this test.
Checkpoints live in `outputs/.cache/shuffled_fdp_test/permutation_v1`; the p-value
overview is replaced only after all 32 comparisons complete. The private
`permutation_v1/p_value_comparison.csv` compares the new permutation p-value with
the previous binomial/normal approximation using the same 999 shuffled counts;
no additional comparison plots or historical reference copies are generated.
The p-value overview overlays the old approximation in translucent bars behind
the permutation bars, with a shared scale linear through 3 and logarithmic above.

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

## Research agents

`AGENTS.md` coordinates four project-local specialists in `.codex/agents/`:
researcher, scientific critic, research planner, and research engineer. They
collaborate in the order the task needs, including consultations and revisions;
the main chat synthesizes their work. The planner specifies the science and
outputs; the engineer owns implementation. Scientific plans require user approval,
and completed work does not automatically start another experiment.
