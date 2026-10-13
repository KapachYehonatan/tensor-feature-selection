#!/usr/bin/env python3
"""Penalty for unstable conditional feature selection on training resamples."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/resampled_stability_batch_16"
GRID={"l1_stability":[.01,.10,1],"size_l1_stability":[.01,.10,1]}

if __name__=="__main__":
    joint.OUTPUT,joint.GRID=OUTPUT,GRID
    joint.main()
