#!/usr/bin/env python3
"""Test a logarithmic coverage cost with the original effect association score."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/effect_log_coverage_batch_21"
GRID = {"effect_log_coverage": [0, .025, .05, .10, .25, .50]}

if __name__ == "__main__":
    joint.OUTPUT, joint.GRID = OUTPUT, GRID
    joint.main()
