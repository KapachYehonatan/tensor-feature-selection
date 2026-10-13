#!/usr/bin/env python3
"""Complementarity after reliable aggregation, with joint training-CV K choice."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/size_correlation_batch_13"
GRID={"size_plus_corr": [0,.05,.10,.25]}

if __name__=="__main__":
    joint.OUTPUT,joint.GRID=OUTPUT,GRID
    joint.main()
