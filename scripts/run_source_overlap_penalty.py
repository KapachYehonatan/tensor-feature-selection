#!/usr/bin/env python3
"""BH score penalty for repeated taxon provenance, not bacterial abundance."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search

search.OUTPUT=PROJECT/"outputs/posthoc_penalty_search/source_overlap_batch_8"
search.GRID={"bh_source_overlap":[0,.5,1,2,4]}

if __name__=="__main__":search.main()
