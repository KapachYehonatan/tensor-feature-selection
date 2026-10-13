#!/usr/bin/env python3
"""Preserve BH at zero and blend toward linear coverage reliability rank."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/bh_coverage_rank_batch_25"
GRID = {"bh_coverage_rank_blend": [0, .25, .50, .75, 1]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
