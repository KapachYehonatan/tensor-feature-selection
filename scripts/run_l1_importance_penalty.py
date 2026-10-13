#!/usr/bin/env python3
"""Penalize weak conditional importance without changing the prediction model."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/l1_importance_batch_14"
GRID={"l1_importance":[.01,.10,1]}

if __name__=="__main__":
    joint.OUTPUT,joint.GRID=OUTPUT,GRID
    joint.main()
