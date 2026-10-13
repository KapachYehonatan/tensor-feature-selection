#!/usr/bin/env python3
"""Favor effects consistent across training subsets, weighted by coverage."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/coverage_partition_batch_36"
GRID = {"coverage_partition_power": [0, .05, .10, .25, .50, 1, 2]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
