#!/usr/bin/env python3
"""Transfer the fixed relative-coverage rule with training-only lambda/K choices."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_size_source_transfer as transfer

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/taxon_coverage_transfer"
GRID={"taxon_coverage":[0,.10,.25,.50,1]}

if __name__=='__main__':
    transfer.OUTPUT,transfer.GRID=OUTPUT,GRID
    transfer.main()
