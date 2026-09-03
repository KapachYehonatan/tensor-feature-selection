# Research plan

## Main question

Determine whether binary outcome signal is carried by bacteria marginals, gene
marginals, or pair-specific abundance remaining after both marginals are
accounted for.

## Method

1. Remove invalid rows and pairs present in fewer than
   `max(5, ceil(0.05 * training samples))` samples.
2. Normalize each sample to unit retained mass and apply a square-root
   transformation for similarity calculations.
3. Use an exact sparse prefix search to assign bacteria and genes to
   high-similarity representatives. Cluster count is the resulting coherent
   representative groups plus singletons.
4. Sum cluster-member marginals and screen bacteria and gene clusters in the
   training split.
5. Fit a ridge-logistic marginal model and calculate out-of-fold outcome
   residuals within the training data.
6. Score every pair against the outcome residual after removing its learned
   co-cluster block aggregate, retain two candidates per block, and use sparse
   logistic selection on these residual pair features.
7. Compare held-out prediction before and after adding residual pairs.

## Experiments

| Experiment | Comparison | Interpretation |
| --- | --- | --- |
| Prevalence | raw versus retained pairs and mass | initial information loss |
| Marginals | intercept, bacteria, genes, both | signal at each dimension |
| Clustering | planted versus recovered clusters in synthetic data | recovery and aggregation fidelity |
| Pairs | pairs alone, marginals, and marginals plus residual pairs | pair signal and its added value beyond marginals |
| Holdout | fixed stratified 80/20 split | fast exploratory prediction estimate |
| Stability | final repeated folds, if justified | reproducibility of selected entities |
| Synthetic blocks | exact versus dropout/member noise | cluster recovery and aggregation loss |
| Synthetic target | null, marginal, pair-only, marginal plus pairs | attribution and false discoveries by level |
| Synthetic scale | 150/250/350/500 samples over seven seeds | recovery, stability, runtime, and peak memory |
| Top-K baselines | pairs, collapsed marginals, clustered marginals, and residual pairs | prediction versus feature budget |

## Synthetic design

The main sparse generator uses 50 bacteria, 10,000 genes, 10 balanced bacteria
clusters, and 100 balanced gene clusters. Each bacteria-cluster × gene-cluster
block has 1%, 5%, 10%, 20%, or 40% prevalence. Shared block detection induces
cluster structure; 5% member dropout and log-normal member noise make members
similar rather than identical and leave approximately 90% zeros overall.

The tensor is generated before the target. Marginal and full targets are
Bernoulli draws from exact logistic models of standardized square-root relative
abundance. They use three bacteria clusters, four gene clusters, and four
mixed-direction pair residuals with distinct entities outside the marginal
signal clusters. Each pair predictor is its square-root relative abundance
after removing the true co-cluster block aggregate. This makes it information
not shared by hundreds of block neighbors. There is no class-dependent
modification of the tensor and no target-performance generation gate.

Evaluation reports raw and prevalence-filtered cluster ARI/NMI, cluster counts,
singletons, exact-comparison diagnostics, PPJoin recall on tractable subsets,
holdout prediction, level-specific precision/recall/F1, null false discoveries,
signal retention after prevalence filtering, oracle target performance,
prevalence quantiles, runtime, and peak RSS. Bacteria and gene selection are
scored on underlying members of planted clusters. The pair-only scenario
separates pair recovery from failures in marginal nuisance estimation.

## Progression gates

- Marginal signal: at least `0.030` holdout AUROC improvement over no-feature
  prediction; confirmation across folds is deferred until the method is stable.
- Clustering: synthetic assignments recover the planted high-similarity blocks
  and every member satisfies both thresholds against its representative.
- Pair signal: recall above `0.50`, useful precision, and at least `0.020`
  AUROC or 10% relative average-precision improvement beyond the marginal model
  in at least one reviewed strong-signal setup.
- Stop before approximate graph search or factorization if both WirbelJ_2018
  and QinJ_2012 fail these simple gates.

## Deliberate limitations

- Representative clustering does not enumerate every similar feature pair. A
  feature that resembles a non-representative but not the representative stays
  separate so that weak chains cannot form.
- Residual tests are exploratory and rely on simple linear/logistic nuisance
  models.
- The supervised two-per-block shortlist and fixed sparse-logistic penalty do
  not provide post-selection p-values. Held-out prediction is valid for the
  fixed split; entity-level error rates remain exploratory.
- The current pair coefficient is deliberately strong (`±5`) to establish an
  identifiable recovery regime before testing weaker effects.
- Cluster components describe predictive redundancy, not biological pathways
  or causal groups.

## Top-K comparison

Use the six cross-sectional real datasets and synthetic `full_signals`. Compare
ten K values from 50 through 600 with one shared training-only 80/20 holdout.
Report AUROC, precision, recall, F1, biological feature counts, and BH RSP where
defined. Average synthetic seeds within sample size before giving sample sizes
equal weight. The scientific pipeline has no fixed 200-feature cap; K limits
only each comparison model, not the underlying selection output.
