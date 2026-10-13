#!/usr/bin/env python3
"""Transfer the frozen training-subset consistency rule to real datasets."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_size_source_transfer as transfer

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/coverage_partition_transfer"
GRID = {"coverage_partition_power": [0, .05, .10, .25, .50, 1, 2]}

if __name__ == "__main__":
    transfer.OUTPUT, transfer.GRID = OUTPUT, GRID
    transfer.main()
