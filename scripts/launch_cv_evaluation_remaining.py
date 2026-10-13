#!/usr/bin/env python3
"""Same resource supervision for the six-cohort extension; separate log/PID files."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import launch_cv_evaluation as launcher

launcher.PREFIX = 'cv_evaluation_remaining'
launcher.RUNNER = PROJECT/'scripts/run_cv_evaluation_remaining.py'
launcher.STUDY_OUTPUT = PROJECT/'outputs/cv_evaluation/remaining'

if __name__ == '__main__': launcher.main()
