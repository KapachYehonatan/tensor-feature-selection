#!/usr/bin/env python3
"""Resume unchanged selectors, sampling RAM/CPU and recording the child exit."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import fcntl
import json
import os
import resource
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
SAMPLE_SECONDS = 5
PREFIX = 'test_free_feature_selection_resume'
COMPLETED = ('HMP_2019_ibdmdb_unique', 'QinJ_2012')


def watch(child, path):
    started = time.monotonic()
    peak_sampled_rss = 0
    with path.open('a', newline='') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['utc_time', 'pid', 'rss_kib', 'hwm_kib', 'swap_kib', 'threads', 'cpu_seconds'])
        while child.poll() is None:
            try:
                proc = Path('/proc')/str(child.pid)
                values = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines() if ':' in line)
                stat = (proc/'stat').read_text().rpartition(')')[2].split()
                cpu = (int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
                rss, hwm, swap, threads = [int(values.get(key, '0').split()[0]) for key in ('VmRSS', 'VmHWM', 'VmSwap', 'Threads')]
                peak_sampled_rss = max(peak_sampled_rss, rss)
                writer.writerow([datetime.now(timezone.utc).isoformat(), child.pid, rss, hwm, swap, threads, cpu])
                handle.flush()
            except (FileNotFoundError, ProcessLookupError):
                pass  # Child may exit between the status and stat reads.
            time.sleep(SAMPLE_SECONDS)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(pid=child.pid, return_code=child.returncode,
                termination_signal=-child.returncode if child.returncode < 0 else None,
                wall_seconds=time.monotonic()-started, cpu_user_seconds=usage.ru_utime,
                cpu_system_seconds=usage.ru_stime, peak_rss_kib=usage.ru_maxrss,
                peak_sampled_rss_kib=peak_sampled_rss,
                resource_note='Linux child peak RSS from wait/reaping; CPU totals include reaped child descendants; periodic samples survive supervisor interruption')


def main():
    logs = PROJECT/'logs'
    with (logs/f'{PREFIX}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protected = [PROJECT/'outputs/test_free_feature_selection'/d for d in COMPLETED]
        for root in protected:
            if json.loads((root/'result.json').read_text())['status'] != 'complete':
                raise ValueError(f'Refusing to resume: completed dataset marker missing: {root}')
        states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for root in protected for p in sorted(root.rglob('*')) if p.is_file()]
        env = os.environ.copy()
        env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        child = subprocess.Popen([str(PROJECT/'.venv/bin/python'), '-u', str(PROJECT/'scripts/run_test_free_feature_selection.py')],
                                 cwd=PROJECT, env=env)
        (logs/f'{PREFIX}.pid').write_text(f'{child.pid}\n')
        (logs/f'{PREFIX}.supervisor.pid').write_text(f'{os.getpid()}\n')
        print(f'RESUME worker PID={child.pid}; supervisor PID={os.getpid()}; HMP/QinJ are excluded through existing completed-cache reuse', flush=True)
        summary = watch(child, logs/f'{PREFIX}.resources.csv')
        summary['completed_datasets_unchanged'] = states == [(name, Path(name).stat().st_size, Path(name).stat().st_mtime_ns) for name, _, _ in states]
        summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (logs/f'{PREFIX}.exit.json').write_text(json.dumps(summary, indent=2))
        print('RESUME EXIT '+json.dumps(summary), flush=True)
        if not summary['completed_datasets_unchanged']:
            raise RuntimeError('Completed HMP/QinJ files changed unexpectedly')
        sys.exit(child.returncode if child.returncode >= 0 else 128-child.returncode)


if __name__ == '__main__': main()
