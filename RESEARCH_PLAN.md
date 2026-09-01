# Research plan

## Main question

Determine whether binary outcome signal is carried by bacteria marginals, gene
marginals, or pair-specific abundance remaining after both marginals are
accounted for.

## Method

1. Remove invalid rows and pairs present in fewer than
   `max(10, ceil(0.05 * training samples))` samples.
2. Normalize each sample to unit retained mass and apply a square-root
   transformation for similarity calculations.
3. Use an exact sparse prefix search to assign bacteria and genes to
   high-similarity representatives. Cluster count is the resulting coherent
   representative groups plus singletons.
4. Sum cluster-member marginals and screen bacteria and gene clusters in the
   training split.
5. Fit a ridge-logistic marginal model and calculate out-of-fold outcome
   residuals within the training data.
6. Screen pair features against the outcome residual, residualize shortlisted
   pairs against selected marginals, and test the pair residuals.
7. Compare held-out prediction before and after adding residual pairs.

## Experiments

| Experiment | Comparison | Interpretation |
| --- | --- | --- |
| Prevalence | raw versus retained pairs and mass | initial information loss |
| Marginals | intercept, bacteria, genes, both | signal at each dimension |
| Clustering | unclustered versus clustered marginals | redundancy reduction without signal loss |
| Pairs | marginals versus marginals plus residual pairs | joint information beyond marginals |
| Holdout | fixed stratified 80/20 split | fast exploratory prediction estimate |
| Stability | final repeated folds, if justified | reproducibility of selected entities |
| Synthetic | bacteria-only, gene-only, pair-only | attribution to the correct level |

## Progression gates

- Marginal signal: at least `0.030` holdout AUROC improvement over no-feature
  prediction; confirmation across folds is deferred until the method is stable.
- Clustering: no more than `0.020` AUROC below unclustered marginals while
  reducing modeled features by at least 20%.
- Pair signal: at least `0.020` AUROC or 10% relative average-precision
  improvement beyond the marginal model.
- Stop before approximate graph search or factorization if both WirbelJ_2018
  and QinJ_2012 fail these simple gates.

## Deliberate limitations

- Representative clustering does not enumerate every similar feature pair. A
  feature that resembles a non-representative but not the representative stays
  separate so that weak chains cannot form.
- Residual tests are exploratory and rely on simple linear/logistic nuisance
  models.
- Cluster components describe predictive redundancy, not biological pathways
  or causal groups.
