#!/usr/bin/env python3
"""Preprocess and run feature selection for one or more binary datasets."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import subprocess
import sys


PROJECT = Path("/home/kapachy/Microbium/tensor_feature_selection")
DATASETS = "all"  # Use one name or a list such as ["WirbelJ_2018", "QinJ_2012"].
MAX_PARALLEL_DATASETS = 3
N_CPUS = 24  # Total CPUs shared by all concurrently running datasets.
POSITIVE_LABELS = {
    "HMP_2019_ibdmdb_unique": "IBD",
    "JieZ_2017": "ACVD",
    "LiJ_2017": "hypertension",
    "NielsenHB_2014_unique": "IBD",
    "QinJ_2012": "T2D",
    "RubelMA_2020": "STH",
    "WirbelJ_2018": "CRC",
    "ZhuF_2020": "schizophrenia",
}
STEPS = [
    "preprocessing/run_preprocessing.py",
    "feature_selection/run_feature_selection.py",
]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def datasets_to_run(selection):
    datasets = list(POSITIVE_LABELS) if selection == "all" else (
        [selection] if isinstance(selection, str) else list(selection)
    )
    unknown = [dataset for dataset in datasets if dataset not in POSITIVE_LABELS]
    if not datasets or unknown:
        raise ValueError(f"Choose one or more configured datasets; unknown: {unknown}")
    return datasets


def cpu_allocation(dataset_count, available_cpus):
    """Return concurrent jobs, native threads per job, and the CPUs to use."""
    if MAX_PARALLEL_DATASETS < 1 or N_CPUS < 1:
        raise ValueError("MAX_PARALLEL_DATASETS and N_CPUS must be positive integers")
    available_cpus = list(available_cpus)
    if N_CPUS > len(available_cpus):
        raise ValueError(
            f"N_CPUS={N_CPUS}, but this process can use only "
            f"{len(available_cpus)} CPUs"
        )
    workers = min(MAX_PARALLEL_DATASETS, N_CPUS, dataset_count)
    return workers, max(1, N_CPUS // workers), available_cpus[:N_CPUS]


def run_dataset(number, total, dataset, threads_per_dataset):
    """Run one dataset's steps sequentially, in its own child process and log."""
    environment = os.environ.copy()
    environment["DATASET"] = dataset
    environment["POSITIVE_LABEL"] = POSITIVE_LABELS[dataset]
    environment["PYTHONUNBUFFERED"] = "1"
    for variable in THREAD_VARIABLES:
        environment[variable] = str(threads_per_dataset)

    log_directory = PROJECT / "logs"
    log_directory.mkdir(exist_ok=True)
    log_path = log_directory / f"{dataset}.log"
    pid_path = log_directory / f"{dataset}.pid"

    with log_path.open("w", encoding="utf-8") as log:
        for step in STEPS:
            command = [sys.executable, str(PROJECT / step)]
            print(f"Running {step}", file=log, flush=True)
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            print(
                f"[{number}/{total}] {dataset}: {step}; PID {process.pid}; "
                f"log {log_path.relative_to(PROJECT)}",
                flush=True,
            )
            return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f"{dataset}: {step} failed with exit code {return_code}; "
                    f"see {log_path}"
                )
    print(f"[{number}/{total}] {dataset}: completed", flush=True)


def main():
    datasets = datasets_to_run(DATASETS)
    if hasattr(os, "sched_getaffinity"):
        available_cpus = sorted(os.sched_getaffinity(0))
    else:
        available_cpus = list(range(os.cpu_count() or 1))
    workers, threads_per_dataset, selected_cpus = cpu_allocation(
        len(datasets), available_cpus
    )
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, selected_cpus)

    print(
        f"Runner PID {os.getpid()}; datasets={len(datasets)}; "
        f"parallel={workers}; total CPUs={N_CPUS}; "
        f"native threads per dataset={threads_per_dataset}",
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_dataset, number, len(datasets), dataset, threads_per_dataset
            ): dataset
            for number, dataset in enumerate(datasets, start=1)
        }
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{dataset}: {error}")
                print(f"FAILED: {failures[-1]}", flush=True)

    if failures:
        raise RuntimeError("Dataset failures:\n" + "\n".join(failures))
    print("\nAll requested datasets completed.", flush=True)


if __name__ == "__main__":
    main()
