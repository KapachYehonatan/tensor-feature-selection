#!/usr/bin/env python3
"""One globally fixed power extension after alignment CV selected power 2."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/coverage_alignment_batch_32"
GRID = {"coverage_alignment_power": [0, .05, .10, .25, .50, 1, 2, 4, 8]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
