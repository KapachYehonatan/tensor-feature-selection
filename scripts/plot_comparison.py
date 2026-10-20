"""Regenerate comparison figures from completed results, without scientific fits."""
from pathlib import Path
import hashlib
import json
import os
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cache import (PROJECT,CACHE,CV,FULL,NATIVE_CV,NATIVE_FULL,
                               DATASETS,METHODS,NATIVE,GRIDS,METRICS,complete,seal,file_hash)
from metrics.correlation import average_pairwise_absolute_correlation

OUTPUT=PROJECT/'outputs'
TABLES=CACHE/'plot_data'
COLORS=dict(zip(METHODS,['#F58518','#B279A2','#4C78A8','#228833','#CC6677','#AA4499','#44AA99']))
STAT_ORDER=[(m,c) for m in list(METHODS)[:3] for c in ('bh','by')]+[(m,'native') for m in NATIVE]
STAT_LABELS=['Bac\nBH','Bac\nBY','Gene\nBH','Gene\nBY','Join\nBH','Join\nBY','SHOE',
             'MW\neBH','Hurdle\nproduct','Hurdle\naverage']


def collect_tables():
    means,spreads,statistics,correlations=[],[],[],[]
    for dataset in DATASETS:
        for root in (CV,NATIVE_CV):
            directory=root/dataset
            if not all((directory/f'fold_{f}/complete.json').exists() for f in range(5)): continue
            mean=pd.read_csv(directory/'mean_fold_metrics.csv',float_precision='round_trip').assign(dataset=dataset)
            sd=pd.read_csv(directory/'sd_fold_metrics.csv',float_precision='round_trip').assign(dataset=dataset)
            methods=list(METHODS)[:3] if root==CV else NATIVE
            expected={(m,k) for m in methods for k in sorted(set(sum(GRIDS.values(),[])))}
            for frame in (mean,sd):
                if frame.duplicated(['method','requested_k']).any() or set(map(tuple,frame[['method','requested_k']].to_numpy())) != expected:
                    raise ValueError(f'Incomplete/duplicate prediction table: {directory}')
            means.append(mean); spreads.append(sd)
        for root in (FULL,NATIVE_FULL):
            directory=root/dataset
            if not (directory/'complete.json').exists(): continue
            statistics.append(pd.read_csv(directory/'statistics.csv',float_precision='round_trip').assign(dataset=dataset))
            c=pd.read_csv(directory/'pairwise_correlations.csv',float_precision='round_trip').assign(dataset=dataset)
            correlations.append(c[c.selection.eq('all_significant')])
    if not all((means,spreads,statistics,correlations)): raise ValueError('No completed comparison tables')
    TABLES.mkdir(parents=True,exist_ok=True)
    frames=[pd.concat(items,ignore_index=True) for items in (means,spreads,statistics,correlations)]
    for frame in frames[2:]:
        if frame.duplicated(['dataset','method','correction']).any(): raise ValueError('Duplicate statistical comparison')
    # Keep only the calculation fields needed for the published figures and caveats.
    fields=['dataset','method','correction','samples','tested_features','real_discoveries','permutations',
            'mean_shuffled_discoveries','p_value','log_p_value','p_value_underflow','small_expected_counts','status']
    frames[2]=frames[2][[c for c in fields if c in frames[2]]]
    for name,frame in zip(('prediction_comparison','prediction_fold_sd','statistics_comparison','pairwise_comparison'),frames):
        frame.to_csv(TABLES/f'{name}.csv',index=False)
    (CACHE/'shuffled_fdp_test').mkdir(exist_ok=True)
    frames[2].to_csv(CACHE/'shuffled_fdp_test/p_values.csv',index=False)
    return frames


def prediction_figures(mean,sd):
    complete=[d for d in DATASETS if set(mean[mean.dataset.eq(d)].method)==set(METHODS)]
    for suffix,ks in GRIDS.items():
        fig,axes=plt.subplots(2,4,figsize=(24,10),sharex=True,sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            for method,label in METHODS.items():
                curve=mean[mean.dataset.eq(dataset)&mean.method.eq(method)&mean.requested_k.isin(ks)].sort_values('requested_k')
                if curve.empty: continue
                spread=sd[sd.dataset.eq(dataset)&sd.method.eq(method)].set_index('requested_k').reindex(curve.requested_k).auroc.to_numpy()
                ax.plot(curve.requested_k,curve.auroc,color=COLORS[method],label=label,marker='o',markersize=3)
                ax.fill_between(curve.requested_k,np.clip(curve.auroc-spread,0,1),np.clip(curve.auroc+spread,0,1),color=COLORS[method],alpha=.08)
            ax.set(title=dataset.replace('_unique',''),xlabel='Requested K',ylabel='AUROC',xticks=ks,ylim=(0,1.01))
            if dataset not in complete: ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
            ax.grid(alpha=.2)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=4)
        fig.suptitle('Prediction: mean fold AUROC ± descriptive fold SD; Top-K is not FDR-controlled')
        fig.tight_layout(rect=(0,.08,1,.93)); fig.savefig(OUTPUT/f'AUC_overview_k_{suffix}.png',dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        aggregate=mean[mean.dataset.isin(complete)].groupby(['method','requested_k']).auroc.mean().reset_index()
        for method,label in METHODS.items():
            curve=aggregate[aggregate.method.eq(method)&aggregate.requested_k.isin(ks)].sort_values('requested_k')
            ax.plot(curve.requested_k,curve.auroc,color=COLORS[method],marker='o',label=label)
        ax.set(xlabel='Requested K',ylabel='Equal-dataset-weight AUROC',xticks=ks,ylim=(0,1.01),
               title=f'{len(complete)}/8 complete datasets; descriptive means')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(OUTPUT/f'AUC_average_k_{suffix}.png',dpi=150); plt.close(fig)


def statistical_figures(statistics,correlations):
    for field,title in [('negative_log10_p','−log10(p-value)'),('real_discoveries','Native / BH / BY discoveries'),
                        ('average_pairwise_absolute_correlation','Full significant-set average pairwise absolute correlation')]:
        frame=correlations if field.startswith('average_pairwise') else statistics.assign(negative_log10_p=-statistics.log_p_value/np.log(10))
        fig,axes=plt.subplots(2,4,figsize=(24,10),sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            part=frame[frame.dataset.eq(dataset)].set_index(['method','correction'])
            for i,(method,correction) in enumerate(STAT_ORDER):
                if (method,correction) not in part.index: continue
                row=part.loc[(method,correction)]
                if np.isfinite(row[field]): ax.bar(i,row[field],color=COLORS[method],hatch='//' if correction=='by' else None)
                elif field.startswith('average_pairwise'):
                    ax.text(i,.95,'skipped' if row.status=='skipped_size_limit' else 'undefined',rotation=90,
                            va='top',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
            if len(part)!=len(STAT_ORDER): ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
            ax.set(title=dataset.replace('_unique',''),xticks=range(len(STAT_ORDER)),xticklabels=STAT_LABELS,ylabel=title)
            ax.tick_params(axis='x',labelsize=7); ax.grid(axis='y',alpha=.2)
            if field in ('negative_log10_p','real_discoveries'): ax.set_yscale('symlog',linthresh=1)
            else: ax.set_ylim(0,1.05)
            if field=='negative_log10_p': ax.axhline(-np.log10(.05),color='black',linestyle=':')
        caption='; full datasets, internal split only for SHOE'
        if field=='negative_log10_p': caption+='\nShuffled FDP test: exploratory binomial/normal-approximation p-values, not FDR certificates'
        fig.suptitle(title+caption)
        fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(OUTPUT/f'{field}_overview.png',dpi=150); plt.close(fig)


def ranked_pairwise_correlations():
    groups=[]
    for dataset in DATASETS:
        if (FULL/dataset/'complete.json').exists():
            groups.extend((dataset,FULL/dataset/f'{m}_features.npz',[(m,None)]) for m in list(METHODS)[:3])
        directory=NATIVE_FULL/dataset
        if (directory/'complete.json').exists():
            groups.append((dataset,directory/'shoe/features.npz',[('shoe',directory/'shoe/selection.npz')]))
            groups.append((dataset,directory/'raw_pairs/features.npz',
                           [(m,directory/'raw_pairs'/m/'selection.npz') for m in NATIVE[1:]]))
    sources=sorted({p for _,matrix,methods in groups for p in [matrix,*[rank for _,rank in methods if rank is not None]]})
    identity=dict(code=file_hash(__file__),metric=file_hash(PROJECT/'metrics/correlation.py'),
        k=GRIDS['10_20_50_75'],sources=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sources])
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    table,marker=TABLES/'ranked_pairwise_correlations.csv',TABLES/'ranked_pairwise_correlations_complete.json'
    if marker.exists() and json.loads(marker.read_text())['signature']==signature and complete(marker,signature):
        return pd.read_csv(table,float_precision='round_trip')
    rows=[]
    for dataset,path,methods in groups:
        matrix=sparse.load_npz(path).tocsr()
        for method,ranking in methods:
            # Baseline matrices are already in unscreened MW p-value/ID order.
            if ranking is None: order=np.arange(matrix.shape[1])
            else:
                with np.load(ranking) as saved: order=saved['order']
            if (order.shape!=(matrix.shape[1],) or not np.issubdtype(order.dtype,np.integer) or
                    not np.array_equal(np.sort(order),np.arange(matrix.shape[1]))):
                raise ValueError(f'Invalid complete prediction ranking: {path}')
            for k in GRIDS['10_20_50_75']:
                rows.append(dict(dataset=dataset,method=method,requested_k=k,
                    **average_pairwise_absolute_correlation(matrix,order[:k])))
        del matrix
        print(f'Ranked Top-K correlations: {dataset}, {", ".join(m for m,_ in methods)}',flush=True)
    if not rows: raise ValueError('No completed full-dataset rankings')
    TABLES.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows); frame.to_csv(table,index=False); seal(marker,signature,[table])
    return frame


def ranked_correlation_figure():
    frame=ranked_pairwise_correlations(); ks=GRIDS['10_20_50_75']
    fig,axes=plt.subplots(2,4,figsize=(24,10),sharex=True,sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        part=frame[frame.dataset.eq(dataset)]
        for method,label in METHODS.items():
            curve=part[part.method.eq(method)].sort_values('requested_k')
            ax.plot(curve.requested_k,curve.average_pairwise_absolute_correlation,
                    color=COLORS[method],label=label,marker='o',markersize=3)
        if len(part)!=len(METHODS)*len(ks): ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
        ax.set(title=dataset.replace('_unique',''),xlabel='Requested K',xticks=ks,ylim=(0,1.01),
               ylabel='Average pairwise absolute correlation')
        ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=4)
    fig.suptitle('Full-dataset Top-K by prediction feature scores; no significance screening')
    fig.tight_layout(rect=(0,.08,1,.93))
    fig.savefig(OUTPUT/'average_pairwise_absolute_correlation_k_10_20_50_75.png',dpi=150); plt.close(fig)


def main():
    mean,sd,statistics,correlations=collect_tables()
    prediction_figures(mean,sd)
    statistical_figures(statistics,correlations)
    ranked_correlation_figure()


if __name__=='__main__': main()
