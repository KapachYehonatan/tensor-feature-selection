# Robust feature selection in ultra-high-dimensional data

This project studies robust feature selection and multiple testing for sparse
microbiome gene abundances with a **(bacterium, gene)** structure. Methods should
generalize beyond these datasets; the current comparison is not a permanent
restriction on the research.

## Project structure

- `datasets/`: dataset catalog, independent-participant loading and explicit sparse input export.
  `datasets/cMD_downloads/<study>/` holds each study's four source files and is ignored by Git.
- `methods/`: shared preprocessing, representations, corrections and compositional pipelines.
  `graphs/` builds exact Bernoulli edges; `clustering/` fits partitions from saved
  graphs; `selectors/` owns MW ranking, Pearson filtering and fixed three-bin e-values.
- `metrics/`: prediction, pairwise correlation and the Shuffled FDP test.
- `execution/`: shared CV/full-data execution, strict stage caches and resource supervision.
- `reporting/`: plots saved tables only; never fits scientific models.
- `experiments/`: short Python definitions of datasets, methods, metrics and outputs.
- `tests/test_experiment.py`: small scientific and cache-integrity fixtures.

The two current experiments are `main_comparison` (Bacteria-collapsed,
Gene-collapsed, Bernoulli-join and Bins e-value across sixteen datasets) and
`join_correlation_filter` (QinJ/Nielsen plus the eight additions, explicit Pearson cutoff list).
Edit their Python definitions to change components, datasets, grids or outputs.
Preprocessing is shared by all methods within an experiment; graph and Leiden
settings belong to their separate components. Graphs can be reused when only a
selector or Leiden resolution changes.

## Scientific protocol

Preserve supplied finite nonnegative values, without sample normalization, log
or square root. Learn prevalence `max(5, ceil(.05*n))` on fitting samples. Keep
raw constants in group means and exclude constant represented predictors after
aggregation. Means use training-defined membership and denominators.

Prediction uses seed-42 five-fold CV, fine K=1–10 and coarse K={10,20,50,75},
without tuning. Top-K rankings are unscreened; scaling and balanced L2 logistic
regression C=1 are training-only. Mean-fold metrics/SD and pooled OOF metrics are
saved separately. Full-dataset statistics never enter CV prediction.
Here, unscreened means no significance cutoff; filtered join still applies its
training-fitted Pearson cutoff to the predictive ranking.

Baselines use MW/BY at .10 over complete eligible representation families.
Bins e-value uses zero/low/high bins and fixed lambda=1, exact conditional
normalization and native e-BH at .10 over the full supplied raw-pair catalog;
the catalog is defined by the cohort input export, which omits pairs absent
throughout that imported cohort. Pairs excluded later have zero evidence.
Predictive Top-K sets are not FDR certificates.
Filtered join compares each candidate with every accepted feature, then uses
full-family masked-p BY for discoveries; prediction uses a separate unscreened ranking.

The Shuffled FDP test uses the same 999 class-preserving label shuffles across
methods, repeats label-dependent selection and counts ties: `(1+B)/(999+1)`.
It tests exchangeable-label global-null discovery enrichment, not the actual
false-discovery fraction. The saved Normal reference p-value is model-dependent.
Complete-set pairwise correlation is undefined below two features and skipped
above 10,000. These previously explored cohorts remain exploratory evidence.

Graphs start at absolute phi .70, increasing by .05 only for resource guards;
retain the 500-million-edge limit and ordinary Leiden convergence, resolution1,
seed17729. External data and the sibling Bernoulli-join environment are read-only.

## Running and outputs

```bash
.venv/bin/python -m experiments.main_comparison --plot-only
.venv/bin/python -m experiments.join_correlation_filter --threshold 0.8 0.7 0.6 --plot-only
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_experiment
```

Remove `--plot-only` for scientific execution/resumption. Long jobs must use
`nohup` and `python -m execution.resources experiments.<name> [arguments]` inside
a persistent user service with `MemoryMax=200G`, `MemorySwapMax=0`. The supervisor
forwards arguments, limits native threads to one and saves logs/PIDs/resources
under `outputs/<experiment>/.cache/logs/`. Default exact-join/Leiden workers are4/1.

`outputs/main_comparison/` contains only the eight requested comparison PNGs;
`outputs/join_correlation_filter/` contains one pilot PNG. Supporting tables and
stage references are private in each experiment's `.cache/`. Unfiltered inputs
live in `cache/inputs/`; expensive reusable stages live in `cache/stages/`.
Strict signatures include actual stage inputs, settings, kernels and relevant
software. Incompatible stages receive distinct keys; damaged checkpoints fail.
Plotting/docs/tests do not invalidate scientific caches. Missing results are
explicit and plotting cannot trigger computation.
Runs stage tables and figures privately before replacing completed exports.
Scientific or rendering failures retain prior tables, protocols and figures;
live status records the failure. Failed first runs keep reusable stages without
publishing partial reports. Explicit resource-aborted, unsealed graph work can
be restarted after retaining its small abort manifest; completed stages stay strict.

`experiments.extend_datasets` acquires the eight approved cohorts using pinned
cMD/ASAP sources, exports unchanged sparse values into `cache/inputs/`, waits for
active comparison jobs, then runs the expanded main comparison and correlation
filters `.8 .7`. Downloads, checksums, import protocols and an accession-overlap
audit stay in `datasets/cMD_downloads/.sources/`. Converters use the installed base R and Arrow
environments read-only. Launch through `execution.resources`, using
`--log-directory outputs/main_comparison/.cache/logs/extension`. Existing scientific
caches are reused; graph launches retain the RAM guard and use the approved
50-GiB free-disk reserve, recorded separately from scientific cache protocols.
Mann–Whitney shuffles reuse pooled ranks/tie corrections with up to 24
shared-memory workers, each retaining one native numerical thread. The optimized
path is fingerprinted separately and checked against SciPy before extending counts.

## Research agents

`AGENTS.md` describes the five optional specialists in `.codex/agents/`:
researcher, scientific critic, planner, engineer and code reviewer. Use subagents
when requested; the main chat coordinates flexible consultations. The planner specifies science/outputs,
the engineer owns implementation, and the code reviewer challenges engineering
decisions for correctness, simplicity and scientific readability. Scientific
changes require user approval.
