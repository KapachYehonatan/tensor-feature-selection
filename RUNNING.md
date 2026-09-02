# Running the research scripts

Run commands from the project root:

```bash
cd /home/kapachy/Microbium/tensor_feature_selection
mkdir -p logs
```

The commands below avoid `systemd-run`, so they do not require systemd
authentication.

## Project rule for long runs

Any command expected to take more than a few minutes, or whose duration is
uncertain, must run under `nohup`. Redirect its output to `logs/`, save its PID,
and print the PID immediately:

```bash
nohup env PYTHONUNBUFFERED=1 \
  .venv/bin/python path/to/script.py \
  > logs/run_name.log 2>&1 &
long_run_pid=$!
echo "$long_run_pid" | tee logs/run_name.pid
```

The number printed by `tee` is the PID. This rule also applies to benchmarks
started by a coding agent, so closing an IDE or agent session does not discard
hours of progress. Short tests and quick inspections may still run in the
foreground.

## Quick checks

Run the unit tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Generate and evaluate the small synthetic smoke preset:

```bash
.venv/bin/python scripts/generate_synthetic.py
.venv/bin/python scripts/evaluate_synthetic.py
```

These commands use the `active_preset` in `synthetic_config.json`, currently
`smoke`.

## Main synthetic experiment

The main preset is not run automatically. Generate it in the background with:

```bash
nohup env PYTHONUNBUFFERED=1 SYNTHETIC_PRESET=main \
  SYNTHETIC_WORKERS=10 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
  .venv/bin/python scripts/generate_synthetic.py \
  > logs/synthetic_generate.log 2>&1 &
long_run_pid=$!
echo "$long_run_pid" | tee logs/synthetic_generate.pid
```

Only after generation finishes, run evaluation similarly:

```bash
nohup env PYTHONUNBUFFERED=1 SYNTHETIC_PRESET=main \
  SYNTHETIC_WORKERS=10 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
  .venv/bin/python scripts/evaluate_synthetic.py \
  > logs/synthetic_evaluate.log 2>&1 &
long_run_pid=$!
echo "$long_run_pid" | tee logs/synthetic_evaluate.pid
```

After evaluation finishes, export the repeated-run figures in the foreground:

```bash
.venv/bin/python scripts/plot_synthetic_results.py
```

The plotting script averages independent seeds within each sample size and
scenario, shows their standard deviation and individual observations, and
writes PNG/PDF figures plus averaged CSV scores under
`outputs/synthetic_evaluation/main/figures/`.

Generation runs independent `(sample size, seed)` configurations concurrently;
evaluation runs independent `(sample size, seed, scenario)` datasets
concurrently. Both default to five workers and each job runs in a fresh child
process, so its peak RSS is meaningful. Set `SYNTHETIC_WORKERS=10` to run ten at
once, or `=1` for sequential execution. Total CPU and memory are approximately
the per-worker requirements multiplied by `SYNTHETIC_WORKERS`; keep the
numerical-library thread variables small enough to avoid CPU oversubscription.

## CPU control

These variables limit native numerical-library threads. `taskset` restricts
the process to the chosen CPU cores:

```bash
env \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  BLIS_NUM_THREADS=4 \
  taskset -c 0-3 \
  .venv/bin/python preprocessing/run_preprocessing.py
```

Change both `4` and `0-3` together when choosing a different core count. See
the CPU numbers available to the current session with:

```bash
grep '^Cpus_allowed_list' /proc/self/status
```

## Memory and CPU control

Without cgroup permission, `prlimit --as` is the practical per-process memory
fallback. This example allows 32 GiB of virtual address space and four CPU
cores:

```bash
env \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  BLIS_NUM_THREADS=4 \
  taskset -c 0-3 \
  prlimit --as=34359738368 \
  .venv/bin/python preprocessing/run_preprocessing.py
```

Useful byte limits are:

| Limit | `prlimit --as` value |
| --- | ---: |
| 8 GiB | `8589934592` |
| 16 GiB | `17179869184` |
| 32 GiB | `34359738368` |
| 64 GiB | `68719476736` |

`--as` limits virtual address space, not only physical resident memory.
NumPy and SciPy may reserve more virtual memory than they actively use, so
leave headroom. If the limit is reached, allocation can fail with
`MemoryError`. Memory limiting is optional and is not used in the recommended
commands above.

Measure a foreground run, including peak resident memory, with:

```bash
/usr/bin/time -v env \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  BLIS_NUM_THREADS=4 \
  taskset -c 0-3 \
  prlimit --as=34359738368 \
  .venv/bin/python preprocessing/run_preprocessing.py
```

## Background preprocessing with `nohup`

```bash
nohup env \
  OMP_NUM_THREADS=16 \
  OPENBLAS_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 \
  NUMEXPR_NUM_THREADS=16 \
  BLIS_NUM_THREADS=16 \
  .venv/bin/python preprocessing/run_preprocessing.py \
  > logs/preprocessing.log 2>&1 &
long_run_pid=$!
echo "$long_run_pid" | tee logs/preprocessing.pid
```

Closing the terminal will not stop the process. Follow its output with:

```bash
tail -f logs/preprocessing.log
```

Check whether it is still running:

```bash
ps -p "$(cat logs/preprocessing.pid)" -o pid,%cpu,%mem,rss,vsz,etime,cmd
```

Stop it gracefully:

```bash
kill "$(cat logs/preprocessing.pid)"
```

Do not start another run writing to the same dataset output directory until
the previous process has stopped.

## Other project scripts

After preprocessing completes, replace the final script path in the command:

```text
clustering/run_similarity_clustering.py
feature_selection/run_feature_selection.py
```

The selected dataset and research settings remain editable constants near the
top of each script. Feature selection uses a fixed 80/20 holdout and two
co-clustering rounds by default; reduce `MAX_CLUSTER_ROUNDS` to `1` only for a
short exploratory smoke run. Keep `INNER_FOLDS=3` for leakage-safe residuals.

## Parallel real datasets

Set `DATASETS` near the top of `run_datasets.py` to `"all"`, one name, or a
list. The same file has two direct resource controls:

```python
MAX_PARALLEL_DATASETS = 3
N_CPUS = 12
```

`N_CPUS` is the total CPU allowance shared by all dataset processes, not an
allowance per dataset. The runner restricts itself and its children to that
many CPUs and divides native-library threads between concurrent datasets.
Each dataset still runs preprocessing followed by feature selection
sequentially.

Long runs must use `nohup`. The runner writes a separate log and current child
PID to `logs/<dataset>.log` and `logs/<dataset>.pid`:

```bash
nohup env PYTHONUNBUFFERED=1 \
  nice -n 10 \
  .venv/bin/python run_datasets.py \
  > logs/datasets.log 2>&1 &
long_run_pid=$!
echo "$long_run_pid" | tee logs/datasets.pid
```

Monitor the runner and one dataset with:

```bash
tail -f logs/datasets.log logs/JieZ_2017.log
ps -p "$(cat logs/datasets.pid)" -o pid,etime,%cpu,%mem,rss,cmd
ps --ppid "$(cat logs/datasets.pid)" -o pid,etime,%cpu,%mem,rss,cmd
```
