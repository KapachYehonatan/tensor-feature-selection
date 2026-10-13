#!/usr/bin/env python3
"""Composition-adjusted size penalties using cached memberships, not pooling."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_posthoc_penalty_search as search

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/composition_size_batch_10"
GRID = {"gene_diverse_size": [0, .05, .10, .25, .50],
        "homogeneous_size": [0, .05, .10, .25, .50]}

if __name__ == "__main__":
    search.OUTPUT, search.GRID = OUTPUT, GRID
    search.main()
