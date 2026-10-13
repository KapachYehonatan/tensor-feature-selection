#!/usr/bin/env python3
"""Supervise the resumable clustering study and record whole-process-tree resources."""
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
PREFIX = 'bernoulli_clustering_analysis'
SAMPLE_SECONDS = 5
LIMIT_BYTES = 100*1024**3


def process_tree(root):
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rpartition(')')[2].split()
            processes[int(path.name)] = (int(fields[1]), fields)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    selected = {root}
    while True:
        children = {pid for pid, (parent, _) in processes.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    rss = cpu = threads = 0
    for pid in selected & processes.keys():
        fields = processes[pid][1]
        rss += int(fields[21])*os.sysconf('SC_PAGE_SIZE')
        cpu += (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        threads += int(fields[17])
    return len(selected & processes.keys()), rss, cpu, threads


def cgroup_values():
    relative = next((line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')), None)
    if relative is None:
        return {}
    root = Path('/sys/fs/cgroup')/relative.lstrip('/')
    result = {}
    for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events', 'cpu.stat'):
        try:
            result[name] = (root/name).read_text().strip()
        except (FileNotFoundError, PermissionError):
            pass
    return result


def main():
    logs = PROJECT/'logs'
    logs.mkdir(exist_ok=True)
    with (logs/f'{PREFIX}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = os.environ.copy()
        env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        child = subprocess.Popen([str(PROJECT/'.venv/bin/python'), '-u', str(PROJECT/'scripts/run_bernoulli_clustering_analysis.py')],
            cwd=PROJECT, env=env, start_new_session=True)
        (logs/f'{PREFIX}.pid').write_text(f'{child.pid}\n')
        (logs/f'{PREFIX}.supervisor.pid').write_text(f'{os.getpid()}\n')
        print(f'Analysis PID={child.pid}; supervisor PID={os.getpid()}; memory ceiling=100 GiB', flush=True)
        def terminate(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        started, peak_rss, peak_cgroup, killed = time.monotonic(), 0, 0, False
        with (logs/f'{PREFIX}.resources.csv').open('a', newline='') as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(['utc_time', 'root_pid', 'processes', 'tree_rss_bytes', 'live_cpu_seconds', 'threads', 'cgroup_memory_current', 'cgroup_memory_peak'])
            while child.poll() is None:
                processes, rss, cpu, threads = process_tree(child.pid)
                group = cgroup_values()
                current = int(group.get('memory.current', '0'))
                peak = int(group.get('memory.peak', '0'))
                peak_rss, peak_cgroup = max(peak_rss, rss), max(peak_cgroup, peak)
                writer.writerow([datetime.now(timezone.utc).isoformat(), child.pid, processes, rss, cpu, threads, current, peak])
                handle.flush()
                if rss > LIMIT_BYTES:
                    killed = True
                    print('Process-tree RSS exceeded 100 GiB; terminating rather than changing settings', flush=True)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(SAMPLE_SECONDS)
        child.wait()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary = dict(pid=child.pid, return_code=child.returncode, memory_limit_triggered=killed,
            wall_seconds=time.monotonic()-started, cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
            sampled_peak_tree_rss_bytes=peak_rss, sampled_cgroup_peak_bytes=peak_cgroup,
            largest_reaped_process_rss_kib=usage.ru_maxrss, cgroup=cgroup_values(), finished_utc=datetime.now(timezone.utc).isoformat(),
            note='Tree RSS double-counts shared pages; cgroup peak includes page cache and is kernel-accounted; CPU includes reaped descendants.')
        (logs/f'{PREFIX}.exit.json').write_text(json.dumps(summary, indent=2))
        print('Analysis exit: '+json.dumps(summary), flush=True)
        sys.exit(child.returncode if child.returncode >= 0 else 128-child.returncode)


if __name__ == '__main__':
    main()
