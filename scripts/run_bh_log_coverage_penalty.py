#!/usr/bin/env python3
"""Discount cached BH evidence by incomplete source coverage; never pool."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/bh_log_coverage_batch_20"
GRID = {"bh_log_coverage": [0, .25, .50, 1, 2]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
