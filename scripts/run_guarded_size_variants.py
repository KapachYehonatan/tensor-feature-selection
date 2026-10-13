#!/usr/bin/env python3
"""Recover fixed one-SE report, then test the all-fold-consistency safeguard."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_guarded_size_penalty as study

study.CV_CACHE=PROJECT/"outputs/posthoc_penalty_search/guarded_size_batch_6"

if __name__=="__main__":
    for name,consistent in (("guarded_size_batch_6_recovered",False),("consistent_size_batch_7",True)):
        study.OUTPUT=PROJECT/"outputs/posthoc_penalty_search"/name
        study.REQUIRE_CONSISTENT_FOLDS=consistent
        study.main()
