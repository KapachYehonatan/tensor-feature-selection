#!/usr/bin/env python3
"""Penalize rank association unsupported by continuous model-direction signal."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search

search.OUTPUT=PROJECT/"outputs/posthoc_penalty_search/linear_alignment_batch_5"
search.GRID={"linear_alignment": [0,.5,1],"bh_linear_alignment": [0,.5,1]}

if __name__=="__main__":search.main()
