#!/usr/bin/env python3
"""Discount cached evidence by conditional continuous gain and source coverage."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/coverage_logistic_gain_batch_27"
GRID = {"coverage_logistic_gain": [0, .05, .10, .25, .50, 1, 2]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
