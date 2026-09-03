# Tensor feature-selection coding instructions

These instructions apply to all work in this project. Follow the user's current
request when it conflicts with this file.

## Style

- Write local research code for a data scientist, not a reusable framework.
- Use the Ponytail skill for coding work when it is available.
- Prefer linear scripts whose scientific flow is visible from top to bottom.
- Put editable paths, thresholds, folds, repetitions, and seeds near the top.
- Do not add command-line arguments, configuration systems, classes, factories,
  or a Python package without a demonstrated need.
- A little repetition is preferable to an abstraction that hides the analysis.
- Keep comments at scientific section boundaries and explain leakage risks.

## Data and outputs

- Treat `/home/kapachy/Microbium/projects/cMD_downloads` as read-only.
- Start any command expected to take more than a few minutes, or whose duration
  is uncertain, with `nohup` and redirected logging. Save its PID to
  `logs/<name>.pid` and print that PID for the user; do not leave long work
  attached only to an interactive tool session.
- Prevalence filtering, clustering, scaling, and feature selection used for
  evaluation must be learned from training samples only.
- Analyze only datasets with one independent sample per participant.
- Read and save sparse matrices; never create the dense tensor or a complete
  gene-gene similarity matrix.
- Save only required sparse preprocessing artifacts and small CSV summaries.
- Print sample, class, pair, gene, bacteria, and cluster counts.
- Display floats with three decimal places while retaining full precision.

## Scope

- First establish interpretable bacteria, gene, and residual-pair signal.
- Similarity thresholds determine clusters; do not request a cluster count.
- Leave weakly similar features as singletons.
- Do not add factorization, random projection, approximate-neighbor, workflow,
  or distributed-computing dependencies until simple experiments justify them.
- Update README.md and RESEARCH_PLAN.md when behavior or conclusions change.
- Keep SCIENTIFIC_PIPELINE.md synchronized with every change to scientific
  preprocessing, transformations, clustering, selection, modeling, or evaluation.

## Checks

- Fail clearly on invalid matrix orientation, labels, pair identifiers,
  all-zero samples, negative abundances, or invalid thresholds.
- Keep one compact `unittest` module covering the scientific invariants.
