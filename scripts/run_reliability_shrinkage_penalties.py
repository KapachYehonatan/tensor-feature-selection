#!/usr/bin/env python3
"""BH-preserving reliability discounts on cached predictors, without graph work."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_posthoc_penalty_search as search

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/reliability_shrinkage_batch_9"
GRID = {"bh_size_shrinkage": [0, .25, 1, 4, 16],
        "bh_support_shrinkage": [0, .01, .05, .20, 1]}

if __name__ == "__main__":
    search.OUTPUT, search.GRID = OUTPUT, GRID
    search.main()
