#!/usr/bin/env python3
"""One global boundary extension after BH-coverage training CV chose power 2."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/bh_coverage_power_batch_35"
GRID = {"bh_coverage_power": [0, .05, .10, .25, .50, 1, 2, 4, 8]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
