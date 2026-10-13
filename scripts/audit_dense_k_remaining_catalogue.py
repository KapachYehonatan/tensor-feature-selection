#!/usr/bin/env python3
"""Dense-K reporting for batch 26's frozen training-selected prefix."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import audit_dense_k_qinj_penalties as audit

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/dense_k_remaining_catalogue_audit"
FROZEN_METHODS = {"catalogue_remaining_power": "remaining_catalogue_batch_26"}

if __name__ == "__main__":
    audit.OUTPUT, audit.FROZEN_METHODS = OUTPUT, FROZEN_METHODS
    audit.main()
