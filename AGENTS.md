# Research project instructions

The goal is robust feature selection or/and statistical multiple testing in
ultra-high-dimensional, sparse microbiome gene-abundance data with a
two-dimensional (bacterium, gene) structure. We focus on binary target variables, but that's not a necessity. It is preferable to design a method that will fit generic high-dimensional data, and not only microbiome. Current methods are experiments,
not permanent restrictions. Follow the user's current request over this file.

## Delegation and collaboration

- By default, work directly without subagents. Do not spawn or reactivate
  subagents unless the user explicitly requests their use. The main agent may
  research, plan, edit, test, and execute jobs directly.
- When the user requests subagents, the following collaboration rules apply.
- Canonical roles live in `.codex/agents/`: `scientific_researcher.toml`,
  `scientific_critic.toml`, `research_planner.toml`, and `research_engineer.toml`.
  Use named agents when supported. With generic spawning, read the relevant TOML
  and pass its `developer_instructions` explicitly. Disclose unavailable
  delegation rather than pretending an independent review occurred.
- When delegation is requested, choose activity order from the task. Scientific
  questions call for research and critical review; scientific planning uses
  findings and previous results;
  approved implementation, routine exports, styling, and status checks need
  engineering rather than an artificial scientific-planning stage.
- Any agent may consult any other through messaging or the coordinator. Reuse
  threads; allow revisions and loops without a fixed sequence or round limit.
  Only the coordinator creates threads; specialists must not spawn agents.
- Preserve meaningful disagreements. Escalate decisions requiring user judgment,
  new authority, expanded scope, or discussion that stops making progress.
  Do not force consensus or manufacture objections.
- Researcher, critic, and planner are read-only by instruction, not separate
  security sandboxes. The engineer is the sole specialist editing files and
  executing jobs. Briefly announce delegation and its purpose.
- End with the requested task. No autonomous follow-up experiment, permanent
  agent transcript, workflow engine, hooks, or persistent agent-memory system.

## Approval and outputs

- Obtain approval of the scientific plan before implementation. An explicit
  request to implement a specified plan counts. Material scientific revisions
  require renewed approval; ordinary technical choices do not.
- Ask what results and exact outputs are expected when unspecified. Explicitly
  requested deliverables already establish scope; do not ask redundantly.
- Prefer one focused scientific change and minimal interpretable outputs. No
  unsolicited plots, diagnostics, reports, duplicate tables, or speculative
  exports. Keep experiment support under `outputs/<experiment>/.cache/` and shared
  scientific stages under `cache/`; preserve the experiment's requested outputs.
- Inspect the existing hierarchy before adding files. Keep executable code out
  of `outputs/`. Ask for design preferences or propose a concise hierarchy change
  for approval rather than silently restructuring the project.
- Preserve instructions not to run yet. Recommendations do not authorize work.

## Scientific and data safeguards

- Keep CV prediction separate from full-dataset statistical findings. Learn
  filtering, clustering, scaling, and feature selection only from training
  samples for evaluation; never import full-data representations into CV.
- Preserve finite nonnegative supplied sparse values: no sample normalization
  or abundance transformation. Classifier scaling is training-fitted only.
- Use independent participants. Validate orientation, labels, mappings, pair IDs,
  thresholds, constants, and zero-abundance samples; fail clearly on invalid data.
- Preserve training-defined aggregation denominators and constant exclusion.
  Distinguish predictive rankings, discoveries, association, and FDR guarantees.
- Read/save sparse matrices; never create the dense tensor or a complete
  gene-gene similarity matrix. In Bernoulli-join, thresholds determine clusters,
  not a requested cluster count; retain weakly similar features as singletons.
- Treat `/home/kapachy/Microbium/projects/cMD_downloads` as read-only. Leave
  external repositories and installed environments untouched. Preserve unrelated
  working-tree changes.
- Inspect existing caches before recomputing. Keep strict input/settings/code
  signatures and original protocols; never bypass incompatibility or accidentally
  refit an expensive experiment to resolve a cache mismatch.

## Engineering and execution

- Use Ponytail for coding when available. Write simple local research code with
  visible scientific flow, editable settings near entry points, and comments
  explaining scientific boundaries and leakage risks.
- Reuse existing helpers and structure. No new frameworks, configuration systems,
  abstractions, or complex dependencies without demonstrated need and approval.
- Use the existing resource supervisor for long or uncertain-duration jobs:
  `nohup`, redirected logs, saved `outputs/<experiment>/.cache/logs/<name>.pid`,
  and PID reported to the user.
  Preserve the 200-GiB ceiling, one numerical thread, four exact-join workers,
  one Leiden worker, edge guards, and explicit resource-failure reporting.
- Keep focused scientific checks in the existing compact `unittest` module.
  Report actual changes, verification, execution status, and remaining risks.
- Retain necessary sample/class/pair/gene/bacteria/cluster counts; display floats
  to three decimals while saving full precision.
- Keep README concise. Do not recreate `RESEARCH_PLAN.md` or
  `SCIENTIFIC_PIPELINE.md` unless requested.
