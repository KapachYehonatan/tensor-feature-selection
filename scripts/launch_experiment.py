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

PREFIX = 'experiment'
LIMIT_BYTES = 200*1024**3
SAMPLE_SECONDS = 5
RUNNER = PROJECT/'scripts/run_experiment.py'
STUDY_OUTPUT = PROJECT/'outputs/.cache/provenance/run'


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
        print(f'{PREFIX} PID={child.pid}; supervisor={os.getpid()}; hard memory ceiling=200 GiB', flush=True)
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
            saved = json.loads(result.read_text()) if result.exists() else {}
            if saved.get('status', 'running') == 'running':
                saved.update(status='interrupted', resource_summary=summary)
                result.write_text(json.dumps(saved, indent=2))
        print(PREFIX+' exit: '+json.dumps(summary), flush=True)
        sys.exit(child.returncode if child.returncode >= 0 else 128-child.returncode)


if __name__ == '__main__': main()
