# Sparse tensor feature selection

This project asks whether a binary outcome contains predictive information at
three levels of sparse metagenomic gene-family data:

1. bacteria marginals;
2. gene marginals;
3. individual gene-bacterium pairs after accounting for both marginals.

The source data remain read-only in
`/home/kapachy/Microbium/projects/cMD_downloads`. The files store observed
`UniRef90|taxon` pairs as sparse sample-by-pair matrices; the dense
sample-by-bacteria-by-gene tensor is never materialized.

## Environment

The project has its own Python 3.12 virtual environment:

```bash
source /home/kapachy/Microbium/tensor_feature_selection/.venv/bin/activate
pip install -r requirements.txt
```

See [RUNNING.md](RUNNING.md) for CPU limits, memory limits, background `nohup`
runs, logs, monitoring, and stopping a process.

See [SCIENTIFIC_PIPELINE.md](SCIENTIFIC_PIPELINE.md) for the concise mathematical
description of the real-data pipeline and synthetic generation and evaluation.

Use [notebooks/preprocessed_dataset_analysis.ipynb](notebooks/preprocessed_dataset_analysis.ipynb)
for a sparse-safe overview of any completed preprocessing output.

## Research flow

Edit the constants at the top of each script, then run from the project root.

```bash
.venv/bin/python scripts/generate_synthetic.py
.venv/bin/python scripts/evaluate_synthetic.py
.venv/bin/python preprocessing/run_preprocessing.py
.venv/bin/python clustering/run_similarity_clustering.py
.venv/bin/python feature_selection/run_feature_selection.py
.venv/bin/python -m unittest discover -s tests -v
```

For preprocessing and feature selection over one or more datasets, edit
`DATASETS`, `MAX_PARALLEL_DATASETS`, and the total `N_CPUS` near the top of
`run_datasets.py`, then run:

```bash
.venv/bin/python run_datasets.py
```

`DATASETS` accepts `"all"`, one dataset name, or a list of names. `"all"`
means the six configured cross-sectional healthy-versus-disease studies:
JieZ_2017, LiJ_2017, QinJ_2012, RubelMA_2020, WirbelJ_2018, and ZhuF_2020.
Up to three datasets run concurrently by default, while each dataset's two
steps remain sequential. A failed dataset skips its later steps without
stopping the other active datasets. Logs and current child PIDs are written per
dataset under `logs/`. The standalone descriptive clustering script is not run.

### 1. Preprocessing

`preprocessing/run_preprocessing.py` reads one raw Matrix Market study, keeps
only classified `gene|bacterium` rows, aligns binary labels,
and saves a sparse sample-by-pair matrix. It also reports the full-data
prevalence result for auditing. Evaluation recomputes prevalence within every
training fold.

The default pair rule is detection in at least
`max(5 samples, 5% of training samples)`.

### 2. Similarity co-clustering

`clustering/run_similarity_clustering.py` alternates between the two biological
dimensions while preserving samples. An edge joins two features only when
their detection Jaccard similarity is at least `0.800` and their square-root
abundance correlation is at least `0.950`. Every member must pass both tests
directly against its cluster representative, preventing weak similarity chains.

No number of clusters is supplied and singletons are expected. At both scales,
an exact sparse prefix search finds candidate representatives with no Jaccard
false negatives, then computes exact abundance similarity. It stores only
rare-coordinate prefix postings and never constructs a complete similarity
matrix. Features are processed from smallest detection support to largest so
that too-small candidates can be skipped efficiently; feature index breaks
ties deterministically.

### 3. Feature selection

`feature_selection/run_feature_selection.py` uses one fixed stratified 80/20
train/holdout split (seed `42`) for these representations:

- intercept-only reference;
- bacteria clusters;
- gene clusters;
- bacteria and gene clusters;
- selected residual gene-bacterium pairs alone, when any pair is selected;
- selected marginals plus residual gene-bacterium pairs.

The training split uses Mann-Whitney screening with Benjamini-Hochberg
correction and two alternating co-clustering rounds by default.
The pair stage first residualizes the outcome with the selected marginal model.
It then scores every pair after removing its learned bacteria-cluster ×
gene-cluster block aggregate, keeps the two strongest candidates per block,
and uses sparse logistic selection to avoid filling the model with hundreds of
equivalent neighbors from one block. The held-out increment in AUROC and
average precision is the primary evidence for pair-level signal. These are
exploratory predictive results, not causal or formal conditional-independence
claims.

The analysis stops if preprocessing or a training-derived prevalence mask
leaves any sample with no retained abundance.

The holdout run prints its model sizes and writes `fold_model_info.csv`, including
retained pairs, available clusters, and bacteria, gene, pair, and total model
features for every representation. The legacy `repeat` and `fold` columns are
kept as `1` for notebook compatibility, while `evaluation`, `test_fraction`,
and `fold_seed` identify the fixed holdout. `metrics.csv` reports holdout
performance; cluster counts refer to model columns, not underlying members.

## Initial study order

Start with `WirbelJ_2018`: 125 independent samples, 65 controls, 60 CRC cases,
and a comparatively small raw matrix. Continue to the other cross-sectional
studies only if it shows stable marginal or pair-level signal.

## Outputs

Generated files live under ignored `outputs/` directories. Large sparse files
are limited to the prepared matrix and compact index arrays. Reported outputs
are CSV tables for filtering, clusters, holdout metrics, and selected features.
Feature-selection outputs also include split-specific model dimensions. A
single holdout cannot estimate selection stability; that is deferred until a
final cross-validation run is justified.

## Synthetic evaluation

`synthetic_config.json` contains a quick `smoke` preset and the intended
`main` experiment. The generator creates six matched scenarios: exact blocks,
noisy blocks, a null target, marginal signals, pair signals, and marginal plus
pair signals. It first generates one clustered zero-inflated log-normal tensor,
then samples targets from fixed logistic models. Pair targets use standardized
deviations from their true bacteria-cluster × gene-cluster block aggregate, so
the planted information is pair-specific rather than a label for 500 nearly
identical block neighbors. It writes sparse coordinates and complete truth under
`outputs/synthetic/<preset>/`; it never creates a dense tensor. The evaluator
calls the same public clustering and holdout-selection functions used for real
data and writes per-run tables plus `REPORT.md` under
`outputs/synthetic_evaluation/<preset>/`.

`scripts/plot_synthetic_results.py` summarizes the main repeated runs as
co-clustering, marginal selection, pair selection, and held-out prediction
figures. Seed runs are shown as observations and averaged within each sample
size and data scenario. Plot labels distinguish data scenarios from clustering
or selection levels and prediction models; pair false discoveries are shown as
the false discovery proportion rather than a raw count. Prediction quality is
shown as four AUROC panels with one common data-driven y-axis range.

Block prevalences range from 1% to 40%, giving approximately 90% zeros overall
while keeping selected pair signals above the prevalence filter. The main
preset uses 50 bacteria, 10,000 genes, 10 bacteria clusters, 100 gene clusters,
four strong pair signals, sample sizes 150/250/350/500, and seven seeds. It is
intentionally not the default. See [RUNNING.md](RUNNING.md) for the background
commands. Synthetic generation and evaluation use five independent worker
processes by default; set `SYNTHETIC_WORKERS` to change the concurrency.

## Top-K comparison

`scripts/run_method_comparison.py` compares BH-ranked pairs, bacteria averages,
gene averages, co-clustered marginals, and co-clustered marginals plus residual
pairs. It uses the six cross-sectional real datasets and only synthetic
`full_signals`. Expensive method results are cached independently, while
`scripts/summarize_method_comparison.py` rebuilds tables and figures from those
caches. Every per-K result records bacteria, gene, pair, and total columns. The
scientific selector no longer imposes a fixed 200-feature cap.

## Validation completed

The compact unit suite covers the scientific invariants. The current smoke
preset completes all six scenarios, exactly recovers the noiseless block
memberships, obtains 100% PPJoin candidate recall, produces 90.6% zeros, and
retains both planted pairs after training prevalence filtering. It does not
recover those pair signals at `n=40`. The smoke preset validates code paths
only; signal-recovery conclusions require the reviewed main experiment across
sample sizes and seeds.

A limited pre-rerun validation used only `n=350`, seeds `202` and `303`. In the
pair-only scenario, pair recall was `0.750` for both seeds, precision was
`1.000` and `0.600`, and held-out AUROC was `0.813` and `0.746` at the final
`C=0.033` setting. One matched null run selected one pair. These checks justify
a full rerun but do not replace it.

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for hypotheses, comparisons, and stop
conditions. Data provenance is in [data/README.md](data/README.md).
