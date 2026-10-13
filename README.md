# Robust feature selection in ultra-high-dimensional microbiome data

This research project aims to develop robust feature-selection and statistical
multiple-testing methods for ultra-high-dimensional settings. We study sparse
microbiome gene-abundance data, whose features have a two-dimensional
**(bacterium, gene)** structure.

Our current method is **Bernoulli-join**, using mean aggregation. We currently
compare it with the **bacteria-collapsed** and **gene-collapsed** baselines across
eight independent-sample cohorts. These experiments use Mann–Whitney tests with
Benjamini–Hochberg correction (MW/BH); this setup does not limit future methods
or multiple-testing approaches.

## Current experimental pipeline

The three current methods learn prevalence filtering from training samples,
normalize retained pair abundances per sample, average the original pair
abundances, and take the square root once. MW/BH screens each complete feature
family at q≤0.10; ranking
is q, p, then feature ID. Prediction uses training-only scaling and balanced L2
logistic regression, C=1. Fine K is 1–10; coarse K is 5,10,20,35,50,60,75,85,100.
No K tuning is performed in the five-fold CV study.

## Data and environments

The project `.venv` uses `requirements.txt`. Exact graph construction and
Leiden use the existing environment in the sibling `bernoulli-correlation-join`
repository. Do not delete or modify that repository or the read-only source
data in `/home/kapachy/Microbium/projects/cMD_downloads`.

`preprocessing/run_preprocessing.py` creates sparse pair-abundance caches and
independent-sample metadata under `outputs/preprocessing`. Configure DATASET
and POSITIVE_LABEL in its environment. HMP and Nielsen use one day-zero stool
sample per participant. Whole-cohort prevalence masks in these caches are
audit previews only: every evaluation split relearns its own mask.

## Original 80/20 comparison

`scripts/run_method_comparison.py` runs all three methods on the deterministic
seed-42 80/20 split. Dataset selection and scientific settings are constants
near the top of the scripts. Existing valid results and Bernoulli graphs are
reused. The historical join requested absolute phi .70; its resource-dependent
effective threshold is recorded per dataset. This behavior is unchanged.

`scripts/summarize_method_comparison.py` regenerates coarse tables/figures from
completed method outputs. `scripts/plot_fine_k_comparison.py` performs only
cached-predictor fits for fine K. Results remain in `outputs/method_comparison`.

## Five-fold evaluation

`scripts/run_cv_evaluation.py` covers HMP/QinJ; its remaining-datasets companion
covers the other six cohorts. Every fold rebuilds its own filtering, exact
graph, Leiden communities, aggregation, MW/BH and prediction model. No
historical assignments enter these folds. Fixed absolute-phi thresholds are
HMP/Nielsen/Rubel .70, ZhuF .75, and QinJ/JieZ/LiJ/Wirbel .80.

Launch the corresponding `launch_cv_evaluation*.py` supervisor with `nohup`
and log redirection inside a persistent user service with `MemoryMax=200G`
and `MemorySwapMax=0`. The supervisor writes PID and resource records in
`logs`. CV uses one native thread, four join workers and one Leiden worker.
The 500-million-edge guard never silently raises CV thresholds.

Preserved fold outputs live in `outputs/cv_evaluation`, with combined
non-copying dataset views in its `all_real` directory. Seven datasets have
five completed folds. ZhuF remains incomplete: its first fold required
518,029,089 edges, exceeding the graph guard. Cleanup does not resolve it.

Primary results are mean fold metrics with descriptive fold SD, not confidence
intervals. Pooled OOF metrics are reported separately. Peaks over K are
descriptive, and these previously explored datasets remain exploratory.

`scripts/plot_cv_vs_holdout.py` draws cached fine/coarse AUROC panels: solid CV
means and same-color dotted 80/20 curves, with a uniform zoomed y-scale.
Its PNGs are in `outputs/cv_evaluation/all_real/figures`.

## Checks and cache provenance

Run `.venv/bin/python -m unittest discover -s tests`.
Shared scientific calculations are in `research_utils.py`.

The cleanup manifest and verification records use `logs/project_cleanup*`.
Original scientific protocol records remain untouched. An explicit compatibility
record permits only verified historical signatures under unchanged inputs,
cache artifacts and refactored scientific code; incompatible changes still fail.
Obsolete experiments were permanently removed without an archive. Git can
recover tracked code, but not deleted untracked results.
