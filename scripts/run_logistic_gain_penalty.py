#!/usr/bin/env python3
"""Penalize marginal evidence not supported by conditional logistic gain."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search

search.OUTPUT=PROJECT/"outputs/posthoc_penalty_search/logistic_gain_batch_4"
search.GRID={"logistic_gain": [0,.5,1]}

if __name__=="__main__":search.main()
