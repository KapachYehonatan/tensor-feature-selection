# Sparse tensor feature selection

Resume interrupted test-free selection with
`scripts/resume_test_free_feature_selection.py`: the original selector script
is unchanged, completed HMP/QinJ files must remain untouched, and partial
fold/ranking caches are reused. Run the nohup launcher in a user-level systemd
service to separate it from the interactive tool session. Resource samples
(every five seconds), worker/supervisor PIDs and the final exit/signal/peak-RSS
record are saved under `logs/test_free_feature_selection_resume.*`.

`scripts/run_test_free_feature_selection.py` compares five test-free selectors
with bacteria/BH and Bernoulli-join mean/BH on all eight independent-sample
real datasets. New methods are zero-aware mutual information, L2/L1/elastic-net
logistic coefficient ranking, and the explicitly named MI top-2,000 → L1
variant. There is no MW/BH screen, bacterial input, or baseline fallback in
any new selector. All nonconstant cached clusters are considered except in
the named shortlist variant. Clustering is not rerun: full sparse means are
reconstructed and cached separately from the existing BH-screened matrices.
Each method independently chooses selector C/K in the existing isolated
three-fold training CV; the final balanced L2 C=1 classifier is unchanged.
Run with bounded threads and nohup; outputs, status, convergence records,
fine/coarse PNGs and exploratory report go to `outputs/test_free_feature_selection`.
HMP runs first as the small real-data smoke check. Existing results are preserved.

`scripts/compare_current_penalty_ablation.py` creates a cache-only six-method
comparison over all eight real datasets: bacteria-collapsed, Bernoulli-join mean
BH baseline, current penalty, and current penalty without parts 2, 3, or 5.
Part 2 is the training-subset effect discount; part 3 is source coverage;
part 5 is the no-fold-loss safeguard. Each ablation uses its own training-CV
parameter/K choice. Parts 2/3 ablations retain the safeguard; removing part 5
retains ordinary mean-CV selection, including BH as an option. No graphs,
representations, or classifiers are recomputed. Outputs are in
`outputs/current_penalty_ablation`: fine/coarse per-dataset and aggregate PNGs,
eight-panel AUROC overviews, metrics, training decisions, and peak summaries.
These remain exploratory known-holdout comparisons, not independent validation.

`scripts/analyze_partition_discount_ablation.py` isolates the questioned subset
discount. Compare effect-only, subset-discount-only, coverage-only and both,
using identical cached BH candidates/representations, classifier and CV grids.
Lock each variant's raw and fold-guarded choices before outer-test reporting.
Also toggle the discount at each of the two training-nominated coverage powers
with identical K, separating a real component change from different gamma/K
choices or the safeguard. Save dense curves and paired patient-bootstrap
intervals at training-selected anchors under partition_discount_ablation.
Intervals condition on fixed fitted models and do not cover adaptive search.
Subset effects are dependent on the full-data screen; this is not a repeated
independent test, corrected p-value, or proof the original MW test is invalid.

The first ablation replay detected default-CSV rounding of near-tied subset
discounts on HMP. Its failure is preserved; corrected replay uses round-trip
float parsing and exact original-prefix checks under partition_discount_ablation_v2.

The completed ablation finds a QinJ interaction, not a generally essential
discount: at gamma 0.10 its dense peak rises 0.830→0.873, but at gamma 1
both peak at 0.834. At matched gamma 0.10/K75, gain +0.086 has conditional
paired-bootstrap 95% interval [-0.029,0.202]. Best training CV slightly favors
coverage alone (0.560 vs 0.557); effects on other datasets are mixed. Read
partition_discount_ablation_v2/REPORT.md before interpreting the winning
procedure as evidence for the subset term. All 100 tests pass; no pipeline
component or previously evaluated method has been changed.

## Completed exploratory penalty benchmark

The first procedure meeting the requested peak criteria is **training-subset
consistency × source coverage, with a no-fold-loss CV guard** (batch 37).
Over the common dense grid K=1..100, QinJ reaches **0.873 at K18**, versus
bacteria **0.837** and original join **0.777**. The same global procedure
retains/improves all seven other dataset peaks: JieZ +0.001, Nielsen +0.012,
and exact BH retention on HMP/LiJ/Rubel/Wirbel/ZhuF. No CV-selected operating
model loses AUROC relative to BH/own-K in these benchmarks. The unguarded
procedure's Wirbel loss remains recorded; no holdout-based dataset exception
is used. QinJ CV selects K75, whose holdout AUROC is 0.771, not its 0.873 peak.

Score each original join predictor by the weaker of its full-training absolute
MW effect and its weakest direction-aligned effect across three stratified
training subsets, clipped at zero, times count-only catalogue coverage R^gamma.
CV selects gamma/K on a fixed global grid. Apply the penalty only with positive
paired CV gain and no fold loss versus BH at BH's own CV-selected K; otherwise
retain BH. Predictors, final balanced L2 C=1 classifier and graphs are unchanged.
Taxon membership counts enter R, but no bacterial abundance predictor is used.

For a fresh dataset, compute these statistics and member counts from its own
training representations, evaluate the same gamma/K grid within its own isolated
training folds, and call `choose_fold_consistent_settings(..., reference_families=())`.
Use `penalty_statistics`, `taxon_coverage_from_assignments` and `select_penalized`
for the scoring step. No dataset name, stored benchmark decision, old outcome,
or bacterial reference is needed. Training spaces need both classes and enough
samples for three stratified subsets. Existing caches are an experiment reuse
mechanism, not a deployment dependency.

Run `scripts/run_fold_consistent_partition_transfer.py` to reproduce the cached
benchmark, `scripts/audit_dense_k_partition_transfer.py` for frozen dense-K
verification, and `scripts/plot_partition_comparison.py` for the updated 12-method
fine/coarse figures. Results are under
`outputs/posthoc_penalty_search/fold_consistent_partition_batch_37`.
All 98 tests passed; all 24 training folds are isolated, references reproduce,
actual dense probabilities reproduce AUROC, and source states remain unchanged.

These are **adaptive exploratory results, not independent validation or a safety
guarantee for unseen datasets**. QinJ's paired CV gain is [0,0,0.036]; a strict
one-SE guard rejects it. The implemented consistency guard is not a confidence
bound. Peak K is descriptive, never the deployment selection rule. Historical
negative evidence below is preserved.

Batch 36 (`run_coverage_partition_penalty.py`) weights source coverage by
training-subset consistency: clip absolute MW effect to the weakest
direction-aligned effect across three stratified training subsets, then multiply
by `R**gamma`. Reuse the existing statistic, powers 0/.05/.10/.25/.50/1/2 and
joint training-only CV. This is a robustness heuristic, not a confidence bound.
Zero power retains the consistency discount; BH is a separate CV option.
Fresh scoring has no historical-results or dataset-name dependency. Graphs,
predictors and final balanced L2 C=1 classifier remain unchanged.

Batch 36 selected gamma 0.10/K75 (CV 0.557), selected-model holdout 0.771,
and exploratory peak 0.851/K20, exceeding bacteria's peak 0.837. All 96 tests
passed. `run_coverage_partition_transfer.py` tests the identical formula/grid
with each of eight datasets' own training CV choices, all locked before holdout.
`audit_dense_k_partition_penalty.py` reports the frozen QinJ prefix at K=1..100
without changing its power or selecting operating K from known holdout values.

Dense batch-36 reporting reaches 0.873/K18. Unguarded transfer retains or
improves six other peaks but loses 0.013 on Wirbel. A strict one-SE guard would
also reject QinJ, whose paired CV improvement is confined to one fold.
Batch 37 (`run_fold_consistent_partition_transfer.py`) separately tests the
globally fixed no-fold-loss guard: nominate penalty parameter/K by mean CV,
compare against BH's own CV-selected K, and apply only with a positive mean
gain and no validation-fold loss. This consistency heuristic is not a confidence
bound. It uses no dataset names, old settings or held-out outcomes. Record the
strict one-SE result diagnostically without changing the fixed guard threshold.


Batch 34 (`run_bh_coverage_power_penalty.py`) discounts cached BH statistical
evidence: `-log10(p)*R**gamma`, with powers 0/0.05/0.10/0.25/0.50/1/2.
Unlike effect/SE penalties, zero power preserves the exact original BH order
among nonconstant candidates. Joint training-only CV chooses power/K or BH/K;
the fixed classifier and representations stay unchanged. This is a ranking
heuristic, not new p-values or BH certification. Pure fresh-data scoring needs
only training p-values and original pair-count coverage, not old results.

Batch 34 chose gamma 2/K60 (CV 0.562), selected holdout 0.703 and peak
0.817/K10. All 95 tests passed. Dense K=1..100 reporting freezes that ranking;
batch 35 independently extends the global power grid once to 4/8, because
training CV selected its upper boundary. All other components remain fixed.

Both checks completed: extended CV still picks gamma 2/K60, and the dense
peak stays 0.817/K10, below 0.834 and bacteria 0.837. Training Top-10 median
absolute nearest correlation falls from BH 0.830 to 0.405, yet this alone does
not close the prediction gap. No transfer is justified. Both batches reproduce
zero-power baseline CV, reference-free decisions, all reference metrics and
actual evidence-loss exports. Negative evidence is preserved; goal remains open.

Batch 33 (`run_coverage_snr_penalty.py`) tests smooth uncertainty discounting:
`abs(effect)*effect**2/(effect**2+SE**2)*R**gamma`. Zero effect/SE scores zero.
Use training MW placement SE and count-only coverage, with fixed powers
0/0.05/0.10/0.25/0.50/1/2 and joint training-CV power/K or BH/K selection.
This is a reliability heuristic, not a posterior probability or confidence bound.
Fresh-data scoring has no dataset-name, historical-result or bacterial-predictor
dependency; graphs, predictors and final balanced L2 C=1 remain unchanged.

Batch 33 selected power 1/K35 (CV 0.560), with selected-model holdout 0.745
and exploratory peak 0.834/K8. All 94 tests pass. `run_coverage_snr_transfer.py`
applies the identical formula/grid/CV procedure to all eight datasets, locking
each training decision before any holdout. `audit_dense_k_snr_penalty.py`
reports the frozen QinJ ranking at every K=1..100; this cannot retune power/K.

The dense audit retains 0.834/K8 and exactly the earlier coverage method's
first eight IDs. Transfer improves standard-grid peaks on Nielsen by 0.072
and Rubel by 0.029, retains the other peaks, but loses 0.006 on JieZ.
CV-chosen model losses also remain on JieZ/Nielsen/Wirbel. The complete goal
is therefore not achieved. All eight reference-free CV choices, cached input
states, four baseline/reference metrics, and identical QinJ rerun were verified.

Batch 29 (`run_coverage_uncertainty_penalty.py`) tests
`max(0,abs(MW effect)-1.96*SE)*R**gamma`, using the existing training rank-placement
SE and count-only source coverage. The coefficient 1.96 is fixed as heuristic
shrinkage, not a certified confidence bound or new BH test. Freeze powers
0/0.05/0.10/0.25/0.50/1/2 and jointly choose power/K by the same cached training
CV; the classifier remains C=1. Export the actual total effect loss. Zero power
retains uncertainty shrinkage; unchanged BH remains a separate CV option.

Batch 29 selected the upper power boundary 2/K35 and peaked at 0.821/K7.
Batch 30 (`run_coverage_boundary_extension.py`) adds powers 4 and 8 globally
for both uncertainty shrinkage and catalogue completion, whose training CV
also selected power 2. Other settings stay fixed; the coefficient 1.96 is not
retuned. Preserve earlier batches and label known-holdout results exploratory.

The extension retains uncertainty power 2 (peak 0.821); completion chooses
power 8 (peak 0.749). Neither improves the leading result. Batch 31,
`run_coverage_alignment_penalty.py`, combines source coverage with the existing
direction-alignment discount: `min(abs(effect),max(0,sign(effect)*Pearson(x,y)))`
times `R**gamma`. Associations use training data only; the same fixed power/K
CV rule applies. Final predictors/classifier remain unchanged.
Batch 31 selected power 2/K35 and peaked at 0.817/K10. The frozen uncertainty
and alignment settings are audited at every K=1..100; alignment also receives
one global power extension to 4 and 8 (batch 32), driven only by its CV boundary.

The dense audit retains uncertainty/alignment peaks 0.821/0.817; batch 32
still selects alignment power 2/K35. Neither improves the leading QinJ score
or merits transfer. All 93 project tests pass; no graph or classifier changes.

Batch 26 (`run_remaining_catalogue_penalty.py`) tests count-only marginal source
coverage: downweight a candidate by the fraction of its source catalogue already
represented by chosen clusters. Score `abs(effect)*(sum_t w_jt*d_jt*(1-c_t))**gamma`,
w_jt=n_jt/S_j, d_jt=n_jt/N_t, c_t=sum previously chosen d_st. Keep powers
0/0.05/0.10/0.25/0.50/1/2, train-only joint CV and the existing C=1 classifier.
This is not source purity, bacterial abundance or predictor pooling. Zero is
effect order; unchanged BH remains a CV option. First-step coverage equals R.
Batch 26 selected gamma 0.25/K75 and peaked at 0.821/K10 on the standard
grid. `audit_dense_k_remaining_catalogue.py` evaluates its frozen prefix at
every K=1..100; this reports missing K values without tuning on the holdout.

The dense remaining-catalogue peak stays 0.821/K10, so it is not transferred.
Batch 27 (`run_coverage_logistic_gain_penalty.py`) discounts cached MW evidence
by ridge-adjusted conditional continuous logistic gain and source coverage,
using the same power grid/CV. It scores from the empty selected set with a
balanced intercept-only null, then refits a C=1 training scorer as features
are chosen. Final classification/representations remain unchanged. Evidence
scores are heuristics, not new p-values; zero power still changes ranking.

Batch 27 selected gamma 0.10/K7 but peaked at 0.741/K4; it is not transferred.
Batch 28 (`run_catalogue_completion_penalty.py`) tests the opposite selected-set
prior to novelty: give a candidate credit for completing an already represented
source's catalogue. Reliability is `sum_t w_jt*(d_jt+covered_t)`, bounded by one
because original cluster memberships are disjoint. Use the same fixed power/CV
grid; keep predictors/classifier unchanged. Extra correlated predictors may or
may not help: prediction, not lower correlation alone, determines success.

Batch 28 selected gamma 2/K5 (holdout 0.724), but peaked at only 0.735/K1;
neither catalogue mode nor conditional gain improved the existing coverage
peak. No new transfer was run for these weaker candidates. All 91 tests pass;
the QinJ bacteria gap and previous transfer losses remain unresolved.

Batch 23 (`scripts/run_coverage_power_penalty.py`) tests proportional source
reliability: `abs(MW effect)*R**gamma`, where R is the existing training-only
source-count coverage. Freeze gamma 0/0.05/0.10/0.25/0.50/1/2 and select gamma/K
jointly in the same cached three-fold training CV, with unchanged balanced L2
C=1. Zero is effect order, not BH; unchanged BH is a separate CV option.
The standalone scorer needs no dataset ID or old results. Exported penalty
is the actual score loss `abs(effect)*(1-R**gamma)`, not multiplied by gamma
again. Results go under `outputs/posthoc_penalty_search/coverage_power_batch_23`.

Batch 23 selected gamma 1/K35 and peaked at 0.834/K8, matching the logarithmic
result. `scripts/audit_dense_k_qinj_penalties.py` checks every K=1..100 for
the frozen additive/logarithmic/proportional settings, alongside unchanged
join and bacteria. This is reporting only: no weight or operating K is chosen
from the dense holdout curve. Previously evaluated points must reproduce.

The dense audit found no hidden coverage peak: all three stay at 0.834/K8,
bacteria at 0.837/K6, and unchanged join reaches 0.777/K14 (rather than the
coarse-grid 0.762/K20). Batch 24 (`run_coverage_support_penalty.py`) tests
`abs(effect)*(R*Neff/n)**gamma`, using training effective support
`Neff/n=(sum x)**2/(n*sum x**2)` on the existing sqrt-abundance predictors.
Freeze the same gamma grid and select gamma/K by training CV. No new data
representation or historical result enters the scorer. The actual effect loss
is exported as the penalty; constants are excluded and units do not matter.
`effective_sample_support` computes this fraction directly from sparse training
predictors without labels; the concentration statistic reuses the same helper.

Batch 24 chose gamma 0.50/K50 (holdout 0.697), with peak 0.796/K5; it is not
transferred. `scripts/run_coverage_power_transfer.py` instead transfers the
unchanged batch-23 proportional family/grid across all eight real datasets,
selecting power/K independently in each training CV and locking all decisions
before holdouts. No stored dataset-specific settings or holdout fallback is used.

Proportional transfer improves Nielsen/Rubel peaks by 0.079/0.039 but loses
0.005 on JieZ and 0.006 on Wirbel. JieZ selected power zero: effect order is
not BH order. Batch 25 (`run_bh_coverage_rank_penalty.py`) therefore preserves
the exact BH endpoint and blends normalized BH ranks toward normalized ranks
of `abs(effect)*R`, with fixed linear coverage and w=0/0.25/0.50/0.75/1.
Only w/K are training-selected; this is a signed rank-disagreement adjustment,
not a p-value or dataset-specific fallback. All previous outputs are preserved.
Batch 25 selected w=0.25/K100 but peaked at 0.797/K7, below the stronger
coverage variants; it is not transferred. All 89 project tests passed.

`scripts/run_bh_log_coverage_penalty.py` tests batch 20's BH-preserving source
coverage score: `-log10(p) - lambda*(-log10(R))`, where R is the existing
count-only relative source coverage. Freeze lambda 0/0.25/0.50/1/2, select
lambda/K jointly using the existing three cached training folds, and keep the
balanced L2 C=1 classifier and separate bacteria reference. Lambda zero restores
original BH order; nonzero scores are heuristics, not new p-values. Save under
`outputs/posthoc_penalty_search/bh_log_coverage_batch_20`; no graphs or predictors
are rebuilt. Known-holdout outcomes remain exploratory.

Fresh datasets do not need these benchmark results: `bh_log_coverage_scores`
accepts only training p-values, training source coverage and a strength;
`taxon_coverage_from_assignments` computes coverage from that dataset's learned
assignments and pair-to-taxon counts. Relearn the existing representation/BH
screen within its own training folds, select strength/K using the same frozen
grid/CV rule, then refit on its training set. Do not reuse any dataset-specific
strength, K, baseline decision or holdout peak from the benchmark outputs.
Standalone parameter selection calls `choose_joint_settings(cv, grid,
reference_families=())`; only the new dataset's baseline/candidate CV scores
are required. Bacterial reference features or historical reference metrics are
not needed for this decision. Benchmark defaults continue reporting bacteria.

Batch 20 chose lambda 0.25/K10 but peaked at 0.757/K20, below unchanged join
0.762; it is not transferred. `scripts/run_effect_log_coverage_penalty.py`
(batch 21) instead retains absolute MW effect as the association score and
changes only the coverage cost to `-log10(R)`. Freeze lambda 0/0.025/0.05/
0.10/0.25/0.50 and the same joint training-CV procedure. Zero restores effect
ranking, so unchanged BH remains a separate CV option. Both selectors are
tested on fresh synthetic inputs with historical-cache access disabled.

Batch 21 selected the upper strength boundary (0.50/K35), reaching peak
0.834/K8 and selected-model holdout 0.745. Batch 22,
`run_effect_log_coverage_extension.py`, adds 1 and 2 globally to that same grid,
motivated by training CV's boundary choice, not by a held-out K or dataset ID.
Keep the original batch and use the same expanded grid for fresh datasets.

The expanded grid still selected 0.50/K35, preserving batch 21's QinJ result.
`run_effect_log_coverage_transfer.py` applies this one frozen family/grid to
all eight real datasets, selecting parameters independently from each dataset's
training folds and locking all choices before holdout evaluation. It reports
exploratory peaks separately from selected-model performance; no dataset-specific
formula or imported benchmark setting is permitted. All 85 tests passed.

Log-coverage transfer completed: peak gains Nielsen +0.066 and Rubel +0.026,
but JieZ loses 0.007. QinJ remains 0.834/K8 versus bacteria 0.837/K6. Other
peaks are unchanged (training CV retained BH on HMP/LiJ/ZhuF); selected-model
losses remain on JieZ/Wirbel. Thus this standalone rule does not meet the full
goal. See `effect_log_coverage_transfer/peak_transfer_summary.csv` and keep
previous results intact; do not patch failures using known dataset identities.

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
means the eight configured independent-sample healthy-versus-disease studies:
HMP_2019_ibdmdb_unique, JieZ_2017, LiJ_2017, NielsenHB_2014_unique,
QinJ_2012, RubelMA_2020, WirbelJ_2018, and ZhuF_2020.
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

The two `_unique` studies use one exact day-zero stool sample per participant;
day-zero duplicates are resolved by greatest read count and then sample ID.

`feature_selection/run_feature_selection.py` uses one fixed stratified 80/20
train/holdout split (seed `42`) for these representations:

- intercept-only reference;
- bacteria clusters;
- gene clusters;
- bacteria and gene clusters;
- selected partial-BH gene-bacterium pairs alone, when any pair is selected;
- selected marginals plus partial-BH gene-bacterium pairs.

The training split uses Mann-Whitney screening with Benjamini-Hochberg
correction and up to four alternating co-clustering rounds by default.
The pair stage first residualizes the outcome with the selected marginal model.
It then scores every pair after removing its learned bacteria-cluster ×
gene-cluster block aggregate and applies BH across all retained pairs. Pairs
with `q <= 0.10` are selected; there is no candidate cap or L1 selection stage.
The held-out increment in AUROC and average precision is the primary evidence
for pair-level signal. These are exploratory predictive results, not causal or
formal conditional-independence claims.

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
intentionally not the default. Synthetic generation and evaluation use five independent worker
processes by default; set `SYNTHETIC_WORKERS` to change the concurrency.

## Top-K comparison

`scripts/run_method_comparison.py` compares bacteria averages, gene averages,
and co-clustered marginals plus partial-BH pairs. The separately restartable
`scripts/run_bernoulli_correlation_join.py` adds an exact detection-correlation
graph method over pair columns, while `scripts/run_jaccard_join.py` adds the
corresponding literal-support Jaccard baseline. Both start at similarity 0.70
and stream-filter already verified edges at higher thresholds when the graph
exceeds the 500-million-edge Leiden cap or an allocation still fails. Current reports
also include the cached joint pairs-plus-bacteria method, while plain BH pairs
and marginal-only co-clustering are hidden. The common comparison uses the
eight independent-sample real datasets and only synthetic `full_signals`;
Both exact join methods are real-data only. Expensive results are cached independently, while
`scripts/summarize_method_comparison.py` rebuilds tables and figures from those
caches. Every per-K result records bacteria, gene, pair, pair-cluster, and total columns. The
scientific selector no longer imposes a fixed 200-feature cap.

## Validation completed

The compact unit suite covers the scientific invariants. The current smoke
preset completes all six scenarios, exactly recovers the noiseless block
memberships, obtains 100% PPJoin candidate recall, produces 90.6% zeros, and
retains both planted pairs after training prevalence filtering. It does not
recover those pair signals at `n=40`. The smoke preset validates code paths
only; signal-recovery conclusions require the reviewed main experiment across
sample sizes and seeds.

Data provenance is in [data/README.md](data/README.md).

The separate real-data Bernoulli-join variation study lives in
`scripts/run_bernoulli_variations.py` and writes to
`outputs/bernoulli_variations/`. It reuses stored exact graphs; the cache-only
`scripts/summarize_bernoulli_variations.py` rebuilds tables and PNG figures.
The two contracted-projection variants cluster bacteria and genes independently
and contract their normalized graphs for at most four Leiden passes.
Run each variation on RubelMA before expanding to all eight datasets.

The isolated QinJ redundancy study is `scripts/run_qinj_redundancy_analysis.py`.
It writes only to `outputs/qinj_redundancy_analysis/`, learns three independent
inner-fold exact graphs at phi threshold 0.80, and compares correlation-filtered
BH, redundancy-penalized selection, abundance pooling, and L2 controls against
join mean and bacteria-collapsed references. A join+bacteria model is diagnostic.
Run with the project `.venv/bin/python`, under `nohup` with a log and PID file;
the script limits numerical threads and runs graph workers sequentially by fold.
The runner records parameter choices before evaluating the outer holdout and
produces CSV summaries, feature provenance, and PNG plots. Existing main-study
outputs are preserved. The output directory records the experiment protocol.

The same runner also includes `residual` selection: lowest training p-value
first, then largest correlation between a candidate and its joint-OLS residual
given the selected set. It uses all BH-screened join candidates and ignores
p-values after the first selection. Compatible fold representations are reused
when selectors change; representation changes require a cache-version bump.

Join-only penalty comparisons also include `penalty_mean` (average absolute
correlation), `penalty_r2` (joint-regression R²), and `penalty_residual_target`
(1 minus absolute correlation of the candidate's regression residual with the
training target). Each selects lambda independently in the same three inner
folds over 0/0.10/0.25/0.50; classifiers still use original predictors and C=1.

`scripts/plot_updated_qinj_comparison.py` extends the original QinJ fine-K figure
from cached metrics: omit Jaccard, retain the other five original methods, and
add four join-only penalty selectors plus complete-link predictor pooling.
It writes `outputs/method_comparison/figures/fine_k_real_QinJ_2012_updated.png`
and a matching CSV under `tables/`, preserving the original figure and caches.
Legend names separate representation/clustering from selection, and explicitly
mark BH ranking without a penalty.

`scripts/plot_best_current_qinj_comparison.py` appends relative source-coverage
penalties to those ten methods, separately labeling the original and refined
parameter grids. It saves fine/coarse `_best_current_penalty` PNGs and CSVs
under the existing comparison figure/table directories without refitting.
Stars mark coverage's training-selected K; curve maxima remain exploratory.

`scripts/plot_aggregate_coverage_comparison.py` adds both coverage-grid protocols
to the existing eight-real-dataset aggregate, with equal dataset weights and
between-dataset SD shading. Each dataset uses its saved training-CV choice
between coverage and unchanged BH (BH retained on HMP, LiJ, ZhuF). Fine/coarse
PNGs and full-precision metric/mean/SD/choice CSVs use `_aggregate_real_with_coverage`
names under `outputs/method_comparison/{figures,tables}/updated_methods/`.
Existing ten-method aggregate metrics are checked and preserved; no refitting.

`scripts/run_real_redundancy_comparison.py` extends that ten-method comparison
to all eight original real datasets, with fine and coarse K and equal-dataset
aggregate figures. Run under `nohup` with log/PID files. It reuses completed QinJ
results, fits three independent inner-fold graphs for each remaining dataset,
and tunes each of the four penalties and merging on training CV over K=1..10.
Fold phi thresholds must match each dataset's original effective threshold;
resource-induced threshold changes fail explicitly. Jobs run sequentially.
New caches/status are under `outputs/real_redundancy_analysis/`; PNGs and CSVs
are under `outputs/method_comparison/{figures,tables}/updated_methods/`.
Per-dataset figures appear as jobs complete; aggregate figures require all eight.

Recovery preserves legacy HMP split metadata after exact index/feature validation,
tolerates atomic shard renames in the external disk-size checker, and explicitly
uses an 800-million community edge cap for ZhuF only (still subject to the
100-GiB memory budget). Thresholds stay fixed; no fallback is permitted. Failed
partial joins are moved to `join.failed-N`, not deleted, before that fold rebuilds.

Recovery is complete for HMP, RubelMA, and ZhuF. All eight datasets now have
ten-method fine/coarse tables and PNGs, plus equal-dataset aggregate figures.
The cache-only verification rerun is logged in
`logs/real_redundancy_verified_rerun.log`; all 23 focused recovery/selection
tests passed. Original baseline metrics and training-fold isolation were checked.

`scripts/run_posthoc_penalty_search.py` starts a separate cache-only search for
score penalties: MW-effect standard error, sparse support, abundance-energy
concentration, small cluster size, association instability across training
partitions, and within-class correlation. It reads only existing screened join
predictors/scores and labels; no graph or preprocessing is run. The first batch
locks strengths on training CV K=5..10 and writes to
`outputs/posthoc_penalty_search/batch_1/`, with the effect-only control and both
fine/coarse exploratory holdout curves. Sparse/empty fold pools are reported.

Completed post-hoc batches and the all-dataset small-cluster transfer check are
summarized in `outputs/posthoc_penalty_search/PROGRESS.md`. The small-cluster
penalty improved QinJ substantially and exceeded bacteria at K=2..4, but degraded
three other datasets; the full predictive/transfer objective remains unachieved.
The separate batch drivers preserve each experiment's outputs and CV choices.

`scripts/run_reliability_shrinkage_penalties.py` tests multiplicative BH-evidence
discounts by original pair count and effective training-sample support. Both
include the unchanged BH order at zero strength; equal cluster sizes also
preserve BH ordering for any size-discount strength. Batch 9 is exploratory,
uses only existing screened fold predictors, and writes separate outputs under
`outputs/posthoc_penalty_search/reliability_shrinkage_batch_9/`.
CV retained BH for size; the support discount lowered QinJ holdout mean K5..10
AUROC to 0.661 versus baseline 0.694. Neither candidate justified transfer.

`scripts/run_composition_size_penalties.py` tests two membership-only adjustments
to the original size penalty: gene repetition and mixed-taxon composition.
It reads cached pair/gene/taxon assignments; no abundances are reaggregated and
no bacterial predictor is used. Batch 10 independently tunes strengths on the
existing training folds and preserves the original BH comparator.

`scripts/run_size_rank_penalties.py` tests rank-disagreement adjustments against
the original small-cluster ordering. Weight zero is BH, not effect-only;
weight one is the small-cluster order for the signed blend, while the capped
version only discounts features whose size rank is worse than their BH rank.
Batch 11 tunes this weight on training CV and retains all fine/coarse outputs.
`scripts/run_size_rank_transfer.py` transfers the signed blend to all eight
datasets with separate training-only weights, locking every choice before
holdout evaluation. `size_rank_transfer/summary.csv` reports fine/large-K
differences and all adverse K; no holdout-driven fallback is permitted.
Transfer improved all other datasets' K5..10 means or retained BH, but lost
fine-grid mean on JieZ/Wirbel and large-K mean on JieZ/Wirbel. It is promising
only partially: the full non-degradation goal is not achieved. All 75 tests pass.

Evaluation update: the user clarified that success concerns the best prediction
over K, not uniform improvement or curve averages. Historical domain losses
remain descriptive, but are not alone disqualifying. The previous rank blend's
QinJ peak is 0.799 (K4), versus join 0.762 (K20) and bacteria 0.837 (K6);
its JieZ peak remains lower than join. `scripts/run_joint_k_penalty_selection.py`
jointly selects weight and K on training CV; join/bacteria select their own K.
Batch 12 reports those models separately from exploratory holdout curve peaks.
QinJ CV chose blend K60, whose holdout AUC was 0.706 despite its 0.799/K4
curve peak. `scripts/run_size_correlation_penalty.py` therefore tests a compound
size/correlation discount with joint training-CV mu/K choice (batch 13), including
unchanged BH as a training-selectable option. No pipeline graph is recomputed.
The compound penalty's peak remained 0.799. Batch 14,
`scripts/run_l1_importance_penalty.py`, instead discounts low conditional
importance from a training-only sparse logistic scorer. It tunes scorer C/K
in CV; the final classifier remains the original balanced L2 C=1 model.
Its QinJ peak was only 0.728. Batch 15,
`scripts/run_size_source_penalty.py`, tests source-overlap discounting after
size-based reliability, using cached membership metadata only. The entire
completed penalty archive was checked over all K; none yet exceeded 0.799.
Batch 15 subsequently reached a new penalty peak of 0.808/K35; bacteria remain
at 0.837/K6. `scripts/run_size_source_transfer.py` transfers this fixed family
with training-only joint parameter/K choices, including the BH option, and
reports peak comparisons separately from CV-selected model comparisons.
Transfer improved Nielsen/Rubel peaks, retained BH on HMP/LiJ/ZhuF, but lowered
JieZ's peak by 0.043 and Wirbel's by 0.006. The QinJ bacteria gap also remains
(0.808 versus 0.837); the objective is still open. All 79 tests pass.

`scripts/run_resampled_stability_penalty.py` tests selection-frequency penalties
from 40 stratified half-samples of each cached training space. A training-only
L1 scorer defines frequency; low frequency is discounted either alone or on
top of the fixed size score. Constant columns are removed before scorer fits,
including resample-specific constants. CV selects scorer C/K; the final model
is still the original L2 classifier. Batch 16 uses no new graphs or predictors.
Resampling reached only 0.790 on QinJ. Batch 17,
`scripts/run_taxon_coverage_penalty.py`, tests relative source coverage rather
than absolute cluster size: candidate members are compared with all retained
pairs from their source taxa, including nonsignificant clusters. This is count
metadata only, not bacterial abundance aggregation or a bacterial predictor.
QinJ's relative-coverage peak reached 0.834/K8 versus bacteria 0.837/K6;
CV selected lambda 0.25/K6 (holdout 0.794). The unchanged family/grid transfers
through `scripts/run_taxon_coverage_transfer.py`, with independent training-only
lambda/K or BH/K choices and separate peak/selected-model reports.
Transfer retained peak AUC on HMP/JieZ/LiJ/Wirbel/ZhuF and improved Rubel, but
lost 0.012 on Nielsen. `scripts/run_taxon_coverage_refinement.py` performs one
global midpoint grid refinement, with the same formula/classifier and joint
training-only lambda/K choice, preserving the original transfer outputs.
The refinement retained/improved every other dataset's peak, but QinJ CV chose
lambda 0.175 and peaked at 0.822, still below bacteria. Batch 19,
`scripts/run_size_coverage_penalty.py`, tests both the fixed size discount and
the relative-coverage discount, with one free coverage strength and joint CV K.
CV chose zero extra coverage strength, reproducing the original size peak
0.799; the combination did not improve the coverage result. The closest
QinJ result remains 0.834 on the original coverage grid; the refined grid
retains/improves all other peaks but reaches only 0.822 on QinJ. Goal still open.

`scripts/run_guarded_size_penalty.py` tests a training-only one-SE safeguard
for the small-cluster penalty, separately for fine and larger K. It retains
original BH ranking when CV gains are uncertain, locks all dataset decisions
before holdout evaluation, and saves guarded/unguarded curves and paired
bootstrap intervals under `outputs/posthoc_penalty_search/guarded_size_batch_6`.
This is an exploratory, budget-dependent selector, not a guarantee of safety.

`scripts/run_bernoulli_clustering_analysis.py` compares extended Leiden optimization,
lower resolution and hierarchical merging of cached Leiden communities, retaining
MW/BH and the original classifier. Run through
`scripts/launch_bernoulli_clustering_analysis.py` with `nohup` and a 100-GiB user
service memory limit; QinJ/HMP pilots precede the other datasets. Results and the
protocol live in `outputs/bernoulli_clustering_analysis`; logs, PID and resource
samples use `logs/bernoulli_clustering_analysis.*`. No Bernoulli graphs are rebuilt.

`scripts/run_no_singletons_analysis.py` removes singleton baseline join clusters
before recomputing MW/BH across the remaining full family, with unchanged cluster
means and classifier. Reuses cached pre-BH matrices and fold-specific assignments;
run with `nohup`, log and PID files. All-eight fine/coarse comparisons against
both BH references and training-CV K choices go to `outputs/no_singletons_analysis`.

`scripts/run_cv_evaluation.py` evaluates bacteria/BH, genes/BH and Bernoulli-join
mean/BH in five full-pipeline, training-isolated folds over all QinJ/HMP samples.
No permanent holdout or K tuning. Launch `scripts/launch_cv_evaluation.py` with
`nohup` inside a user service capped at 200 GiB (`MemoryMax=200G`, no swap).
Fold-local caches, mean-fold/pooled-OOF curves and the report go to
`outputs/cv_evaluation`; logs, PID and resource records use `logs/cv_evaluation.*`.

`scripts/launch_cv_evaluation_remaining.py` runs the other six cohorts with the
same five-fold protocol and 200-GiB service limit, preserving the completed pilot.
New fits live under `outputs/cv_evaluation/remaining`; combined all-eight summaries
and non-copying dataset views live under `outputs/cv_evaluation/all_real`.
Its nohup log/PID/resource prefix is `logs/cv_evaluation_remaining.*`.

`scripts/plot_cv_vs_holdout.py` draws fine/coarse AUROC dataset panels from cached
results: solid mean-fold CV and same-color dotted historical 80/20 curves. PNGs
live in `outputs/cv_evaluation/all_real/figures`; incomplete CV panels are labeled.
