#!/usr/bin/env python3
"""One global boundary extension for two training-CV boundary selections."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/coverage_boundary_batch_30"
POWERS = [0, .05, .10, .25, .50, 1, 2, 4, 8]
GRID = {"coverage_uncertainty_power": POWERS, "catalogue_completion_power": POWERS}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
