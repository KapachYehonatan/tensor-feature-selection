#!/usr/bin/env python3
"""Train the degree of size-rank adjustment with unchanged BH at zero."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_posthoc_penalty_search as search

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/size_rank_batch_11"
GRID = {"bh_size_rank_blend": [0, .25, .50, .75, 1],
        "bh_size_rank_cap": [0, .25, .50, .75, 1]}

if __name__ == "__main__":
    search.OUTPUT, search.GRID = OUTPUT, GRID
    search.main()
