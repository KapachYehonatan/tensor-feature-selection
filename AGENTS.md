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

## Output scope and project organization

- Always ask what results and exact outputs the user expects before implementing
  or running an experiment or analysis. Use explicitly requested deliverables as
  the agreed scope; clarify missing details instead of inventing extra outputs.
- Generate only agreed plots, tables, and reports. Do not add unsolicited
  diagnostics, duplicate summaries, or speculative exports. Keep only necessary
  supporting data and caches privately under `outputs/.cache/`.
- Inspect the existing project directory and hierarchy before adding code or
  artifacts. Reuse the appropriate existing modules and locations; keep code
  out of `outputs/` and keep public outputs limited to requested deliverables.
- When a task raises a project-design or hierarchy decision, ask the user for
  their preferred design or propose a concise recommended change and obtain
  agreement before implementing it. Do not silently introduce new structures.

## Data and outputs

- Treat `/home/kapachy/Microbium/projects/cMD_downloads` as read-only.
- Start any command expected to take more than a few minutes, or whose duration
  is uncertain, with `nohup` and redirected logging. Save its PID to
  `logs/<name>.pid` and print that PID for the user; do not leave long work
  attached only to an interactive tool session.
- Prevalence filtering, clustering, scaling, and feature selection used for
  evaluation must be learned from training samples only.
- Preserve supplied nonnegative sparse values: no sample normalization or
  abundance transformation. Scaling belongs only inside the classifier.
- Analyze only datasets with one independent sample per participant.
- Read and save sparse matrices; never create the dense tensor or a complete
  gene-gene similarity matrix.
- Save only required sparse preprocessing artifacts and small CSV summaries.
- Print sample, class, pair, gene, bacteria, and cluster counts.
- Display floats with three decimal places while retaining full precision.

## Scope

- The research goal is robust feature selection and statistical multiple testing
  in ultra-high-dimensional settings, specifically sparse microbiome
  gene-abundance data with a two-dimensional (bacterium, gene) feature structure.
- The current experiment compares bacteria-collapsed, gene-collapsed and
  Bernoulli-join means with raw-pair SHOE, MW-eBH and Hurdle product/average
  selectors. Prediction uses CV; statistical findings use all samples. Baseline
  prediction is unscreened MW ranking; BH/BY define statistical subsets. This
  setup is not a permanent restriction on methods or statistical tests.
- In the current Bernoulli-join method, similarity thresholds determine clusters;
  do not request a cluster count.
- Leave weakly similar features as singletons.
- Do not add factorization, random projection, approximate-neighbor, workflow,
  or distributed-computing dependencies until simple experiments justify them.
- Keep README.md concise; update it only when usage or behavior changes.
- Do not recreate RESEARCH_PLAN.md or SCIENTIFIC_PIPELINE.md unless explicitly
  requested. Store experiment-specific results in their existing output reports.

## Checks

- Fail clearly on invalid matrix orientation, labels, pair identifiers,
  all-zero samples, negative abundances, or invalid thresholds.
- Keep one compact `unittest` module covering the scientific invariants.
