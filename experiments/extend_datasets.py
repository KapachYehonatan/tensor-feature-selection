"""Prepare eight approved cohorts, then extend the two existing experiments sequentially."""
from pathlib import Path
import fcntl
import time
from datasets.acquire import prepare_all
from execution import pipeline

THRESHOLDS = (.8, .7)
PROJECT = Path(__file__).resolve().parents[1]


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
        return pipeline.run(experiment)


def main():
    failures = prepare_all()
    # Import definitions after preparation/queueing; existing workers keep their original definitions.
    from experiments.main_comparison import EXPERIMENT
    from experiments.join_correlation_filter import experiment
    failures += run_locked(EXPERIMENT)
    failures += run_locked(experiment(THRESHOLDS))
    raise SystemExit(int(bool(failures)))


if __name__ == '__main__':
    main()
