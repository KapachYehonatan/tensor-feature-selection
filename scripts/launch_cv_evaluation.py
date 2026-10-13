#!/usr/bin/env python3
"""Run the sequential CV study inside its hard-capped user service."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import fcntl
import json
import os
import resource
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.launch_bernoulli_clustering_analysis import process_tree, cgroup_values

PREFIX = 'cv_evaluation'
LIMIT_BYTES = 200*1024**3
SAMPLE_SECONDS = 5
RUNNER = PROJECT/'scripts/run_cv_evaluation.py'
STUDY_OUTPUT = PROJECT/'outputs/cv_evaluation'


def main():
    logs = PROJECT/'logs'
    logs.mkdir(exist_ok=True)
    with (logs/f'{PREFIX}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        group = cgroup_values()
        limit = group.get('memory.max', 'max')
        if limit == 'max' or int(limit) > LIMIT_BYTES:
            raise RuntimeError('Launch in a user service with MemoryMax=200G and MemorySwapMax=0')
        env = os.environ.copy()
        env.update({n: '1' for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')})
        child = subprocess.Popen([str(PROJECT/'.venv/bin/python'), '-u', str(RUNNER)],
            cwd=PROJECT, env=env, start_new_session=True)
        (logs/f'{PREFIX}.pid').write_text(str(child.pid)+'\n')
        (logs/f'{PREFIX}.supervisor.pid').write_text(str(os.getpid())+'\n')
        print(f'CV evaluation PID={child.pid}; supervisor={os.getpid()}; hard memory ceiling=200 GiB', flush=True)
        def stop(signum, frame):
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        started, peak_rss, peak_cgroup, killed = time.monotonic(), 0, 0, False
        with (logs/f'{PREFIX}.resources.csv').open('a', newline='') as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(['utc_time', 'pid', 'processes', 'tree_rss_bytes', 'live_cpu_seconds', 'threads',
                                 'cgroup_memory_current', 'cgroup_memory_peak'])
            while child.poll() is None:
                count, rss, cpu, threads = process_tree(child.pid)
                group = cgroup_values()
                peak_rss = max(peak_rss, rss)
                peak_cgroup = max(peak_cgroup, int(group.get('memory.peak', 0)))
                writer.writerow([datetime.now(timezone.utc).isoformat(), child.pid, count, rss, cpu, threads,
                                 group.get('memory.current', ''), group.get('memory.peak', '')])
                handle.flush()
                if rss > LIMIT_BYTES:
                    killed = True
                    print('Process-tree RSS exceeded 200 GiB; terminating without changing science settings', flush=True)
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(SAMPLE_SECONDS)
        child.wait()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary = dict(pid=child.pid, return_code=child.returncode, wall_seconds=time.monotonic()-started,
            sampled_peak_tree_rss_bytes=peak_rss, sampled_cgroup_peak_bytes=peak_cgroup,
            memory_limit_triggered=killed, cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
            cgroup=cgroup_values(), finished_utc=datetime.now(timezone.utc).isoformat(),
            note='Tree RSS double-counts shared pages; cgroup peak includes cache. Native threads=1, join workers=4, Leiden workers=1.')
        (logs/f'{PREFIX}.exit.json').write_text(json.dumps(summary, indent=2))
        if child.returncode:
            import pandas as pd
            status = STUDY_OUTPUT/'status.csv'
            if status.exists():
                rows = pd.read_csv(status)
                running = rows.status.eq('running')
                rows.loc[running, 'status'] = 'failed'
                rows.loc[running, 'error'] = f'Interrupted process exit {child.returncode}; see logs/{PREFIX}.exit.json'
                rows.to_csv(status, index=False)
            result = STUDY_OUTPUT/'result.json'
            if not result.exists(): result.write_text(json.dumps(dict(status='interrupted', resource_summary=summary), indent=2))
        print('CV evaluation exit: '+json.dumps(summary), flush=True)
        sys.exit(child.returncode if child.returncode >= 0 else 128-child.returncode)


if __name__ == '__main__': main()
