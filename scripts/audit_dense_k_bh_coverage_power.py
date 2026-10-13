#!/usr/bin/env python3
"""Dense reporting for frozen BH-coverage and effect-coverage rankings."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import audit_dense_k_qinj_penalties as audit

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/dense_k_bh_coverage_audit"
FROZEN_METHODS = {"bh_coverage_power": "bh_coverage_power_batch_34",
                  "coverage_power": "coverage_power_batch_23"}

if __name__ == "__main__":
    audit.OUTPUT, audit.FROZEN_METHODS = OUTPUT, FROZEN_METHODS
    audit.main()
