#!/usr/bin/env python3
"""Append frozen subset-reliability protocols to the old real comparison."""
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.plot_best_current_qinj_comparison import append_coverage, GRIDS, SEARCH
from scripts.plot_aggregate_coverage_comparison import aggregate
from scripts.plot_updated_qinj_comparison import COMPARISON, METRICS, OLD_METHODS, NEW_METHODS, plot_comparison
import numpy as np
import pandas as pd

OUTPUT = SEARCH / "fold_consistent_partition_batch_37/figures"
TABLES = COMPARISON / "tables/updated_methods"
PROTOCOLS = {
    "partition_unguarded": ("coverage_partition_transfer", "mean-CV choice without fold guard"),
    "partition_consistent": ("fold_consistent_partition_batch_37", "no-fold-loss training guard"),
}
EXTRA = {
    "partition_unguarded":"Bernoulli-join mean | subset consistency × coverage or BH (training CV)",
    "partition_consistent":"Bernoulli-join mean | subset consistency × coverage or BH (fold-consistent training CV)",
}
METHODS = [*OLD_METHODS, *NEW_METHODS, *EXTRA]


def main():
    marker = json.loads((PROJECT/"outputs/real_redundancy_analysis/complete.json").read_text())
    datasets = marker['datasets']
    if marker['status'] != 'complete' or len(set(datasets)) != 8:
        raise ValueError('The original eight-dataset comparison must be complete')
    choices = {}
    for method,(batch,_) in PROTOCOLS.items():
        result=json.loads((SEARCH/batch/'result.json').read_text())
        if result['status'] != 'complete' or not result['source_caches_unchanged']:
            raise ValueError(f'Incomplete or changed-source protocol: {batch}')
        choices[method]=json.loads((SEARCH/batch/'all_training_choices.json').read_text())
        if set(choices[method]) != set(datasets):raise ValueError('Different dataset sets')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for domain,ks in GRIDS.items():
        old_grid='fine_k' if domain=='fine' else 'coarse'
        original=pd.read_csv(TABLES/f'{old_grid}_metrics.csv')
        frames=[]
        for dataset in datasets:
            data=original[original.dataset.eq(dataset)].copy(); labels={}
            for method,(batch,description) in PROTOCOLS.items():
                root=SEARCH/batch/dataset
                metrics=pd.read_csv(root/'holdout_metrics.csv')
                locked=pd.read_csv(root/'locked_parameters.csv')
                choice=choices[method][dataset]
                setting=locked[locked.family.eq(choice['family'])]
                if len(setting)!=1 or setting.iloc[0].strength!=choice['strength'] or setting.iloc[0].requested_k!=choice['requested_k']:
                    raise ValueError(f'Training lock mismatch: {method}/{dataset}')
                data,labels[method]=append_coverage(data,metrics,locked,method,description,ks,
                                                  dataset=dataset,family=choice['family'])
            frames.append(data)
            if dataset=='QinJ_2012':
                plot_comparison(data,f'QinJ_2012 — updated methods ({domain} K)\nExploratory holdout; stars = training-selected K',
                                OUTPUT/f'{domain}_k_QinJ_2012.png',ks,extra_methods=labels)
        data=pd.concat(frames,ignore_index=True)
        mean,spread=aggregate(data,datasets,ks,methods=METHODS)
        old=pd.read_csv(TABLES/f'{old_grid}_aggregate_mean.csv')
        check=mean[mean.method.isin([*OLD_METHODS,*NEW_METHODS])].merge(old,on=['method','requested_k'],suffixes=('_new','_old'),validate='one_to_one')
        if len(check)!=10*len(ks):raise ValueError('Incomplete old aggregate')
        np.testing.assert_allclose(check[[f'{m}_new' for m in METRICS]],check[[f'{m}_old' for m in METRICS]],rtol=0,atol=1e-7)
        plot_comparison(mean,f'All 8 real datasets — equal-weight mean ± between-dataset SD ({domain} K)\n'
                        'Added methods use each dataset’s training-only choices; shaded SD is not a confidence interval\n'
                        'Exploratory known-holdout curves',OUTPUT/f'{domain}_k_aggregate_real.png',ks,
                        errors=spread,extra_methods=EXTRA)
        data.to_csv(OUTPUT/f'{domain}_k_metrics.csv',index=False)
        mean.to_csv(OUTPUT/f'{domain}_k_aggregate_mean.csv',index=False)
        spread.to_csv(OUTPUT/f'{domain}_k_aggregate_sd.csv',index=False)
        print(f'Saved {len(datasets)} datasets × {len(METHODS)} methods × {len(ks)} K values; all old aggregates reproduced',flush=True)


if __name__ == '__main__':main()
