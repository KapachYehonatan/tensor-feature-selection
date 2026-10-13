#!/usr/bin/env python3
"""Extend the small-cluster strength grid because training CV chose its boundary."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search

search.OUTPUT=PROJECT/"outputs/posthoc_penalty_search/cluster_size_boundary_batch_3"
search.GRID={"cluster_size": [0,.05,.1,.25,.5,1]}

if __name__=="__main__":search.main()
