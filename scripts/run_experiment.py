"""Run the seven-method CV and full-dataset experiments sequentially."""
from pathlib import Path
import os
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from experiments import cross_validation, full_dataset
from experiments.cache import PROVENANCE, write_json
from scripts.plot_comparison import main as plot


def main():
    progress=dict(cv='pending',full_dataset='pending',plots='pending')
    for name,run in [('cv',cross_validation.main),('full_dataset',full_dataset.main),
                     ('shuffled_test',full_dataset.run_shuffled_test),('plots',plot)]:
        progress[name]='running'; write_json(PROVENANCE/'run/status.json',progress)
        try: run()
        except BaseException:
            progress[name]='failed'; write_json(PROVENANCE/'run/status.json',progress)
            raise
        progress[name]='complete'; write_json(PROVENANCE/'run/status.json',progress)


if __name__=='__main__': main()
