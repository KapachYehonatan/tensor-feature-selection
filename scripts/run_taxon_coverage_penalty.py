#!/usr/bin/env python3
"""Relative source coverage as a post-hoc reliability penalty, no pooling."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/taxon_coverage_batch_17"
GRID={"taxon_coverage":[0,.10,.25,.50,1]}

if __name__=="__main__":
    joint.OUTPUT,joint.GRID=OUTPUT,GRID
    joint.main()
