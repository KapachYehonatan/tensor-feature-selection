"""Run the configured shuffle budget; never invoke graphs, CV or real-label selection."""
import os
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.full_dataset import run_shuffled_test

if __name__=='__main__': run_shuffled_test()
