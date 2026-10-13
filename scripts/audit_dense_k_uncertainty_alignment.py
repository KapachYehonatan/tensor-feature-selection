#!/usr/bin/env python3
"""Report all K for frozen uncertainty/alignment penalties, without tuning."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import audit_dense_k_qinj_penalties as audit

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/dense_k_uncertainty_alignment_audit"
FROZEN_METHODS = {"coverage_uncertainty_power": "coverage_uncertainty_batch_29",
                  "coverage_alignment_power": "coverage_alignment_batch_31"}

if __name__ == "__main__":
    audit.OUTPUT, audit.FROZEN_METHODS = OUTPUT, FROZEN_METHODS
    audit.main()
