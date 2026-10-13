#!/usr/bin/env python3
"""Report dense K for the frozen partition-reliability rule and benchmark."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import audit_dense_k_qinj_penalties as audit

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/dense_k_partition_audit"
FROZEN_METHODS = {"coverage_partition_power": "coverage_partition_batch_36",
                  "coverage_power": "coverage_power_batch_23"}

if __name__ == "__main__":
    audit.OUTPUT, audit.FROZEN_METHODS = OUTPUT, FROZEN_METHODS
    audit.main()
