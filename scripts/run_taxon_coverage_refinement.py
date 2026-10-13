#!/usr/bin/env python3
"""One global midpoint refinement of the fixed coverage penalty grid."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_size_source_transfer as transfer

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/taxon_coverage_refined_transfer"
# One midpoint in every original [0,.10,.25,.50,1] interval; no K/holdout tuning.
GRID={"taxon_coverage":[0,.05,.10,.175,.25,.375,.50,.75,1]}

if __name__=='__main__':
    transfer.OUTPUT,transfer.GRID=OUTPUT,GRID
    transfer.main()
