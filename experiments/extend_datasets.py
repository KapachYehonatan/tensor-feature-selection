"""Prepare eight approved cohorts, then extend the two existing experiments sequentially."""
from pathlib import Path
import fcntl
import shutil
import subprocess
import time
from unittest.mock import patch
from datasets.acquire import prepare_all
from execution import cache, pipeline

THRESHOLDS = (.8, .7)
FREE_DISK_GIB = 100
PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = subprocess.run


def graph_run(command, **kwargs):
    """Apply the approved disk reserve without changing scientific cache identities."""
    if not isinstance(command, (tuple, list)) or str(PROJECT / 'methods/graphs/worker.py') not in command:
        return ORIGINAL_RUN(command, **kwargs)
    command = list(command)
    command[command.index('--free-disk-gb') + 1] = str(FREE_DISK_GIB)
    required = sum(float(command[command.index(flag) + 1]) for flag in ('--memory-gb', '--memory-reserve-gb'))
    disk = float(command[command.index('--free-disk-gb') + 1])
    while True:
        available = int(next(x for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:')).split()[1]) / 1024**2
        free = shutil.disk_usage(PROJECT).free / 1024**3
        if available < required or free < disk:
            print(f'Waiting for graph resources: RAM={available:.1f}/{required:.1f} GiB; '
                  f'free disk={free:.1f}/{disk:.1f} GiB', flush=True)
            time.sleep(30)
            continue
        try:
            return ORIGINAL_RUN(command, **kwargs)
        except subprocess.CalledProcessError:
            log = kwargs.get('stdout')
            lines = Path(log.name).read_text()[-4096:].strip().splitlines() if log is not None else []
            last = lines[-1] if lines else ''
            if (not (last.startswith('RuntimeError: Need at least ') and last.endswith(' GiB available before starting'))
                    and last != 'RuntimeError: Graph/partition would violate the free-disk reserve'):
                raise
            print('Resources changed during worker startup; waiting before retry.', flush=True)
            time.sleep(30)


def run_locked(experiment):
    logs = PROJECT / 'outputs' / experiment.name / '.cache/logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f'{experiment.name}.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print(f'Waiting for active {experiment.name} job to finish; its outputs remain untouched.', flush=True)
                time.sleep(30)
        with patch.object(subprocess, 'run', graph_run):
            return pipeline.run(experiment)


def main():
    cache.write_json(PROJECT / 'outputs/main_comparison/.cache/logs/extension/resource_override.json',
        dict(declared_free_disk_gib=500, effective_free_disk_gib=FREE_DISK_GIB,
             approved='2026-10-29', reason='Operational disk reserve only; scientific settings and cache checks unchanged.'))
    print(f'Graph/clustering free-disk reserve: {FREE_DISK_GIB} GiB; completed scientific caches remain compatible.', flush=True)
    failures = prepare_all()
    # Import definitions after preparation/queueing; existing workers keep their original definitions.
    from experiments.main_comparison import EXPERIMENT
    from experiments.join_correlation_filter import experiment
    failures += run_locked(EXPERIMENT)
    failures += run_locked(experiment(THRESHOLDS))
    raise SystemExit(int(bool(failures)))


if __name__ == '__main__':
    main()
