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
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from experiments.cache import (PROJECT,CACHE,CV,FULL,NATIVE_CV,NATIVE_FULL,
                               DATASETS,METHODS,NATIVE,GRIDS,METRICS,SHUFFLED,complete,seal,file_hash,
                               validate_fingerprints,SHUFFLE_COUNT)
from metrics.correlation import average_pairwise_absolute_correlation, PAIRWISE_FEATURE_LIMIT

OUTPUT=PROJECT/'outputs'
TABLES=CACHE/'plot_data'
COLORS=dict(zip(METHODS,['#F58518','#B279A2','#4C78A8','#228833','#CC6677','#AA4499','#44AA99']))
STAT_ORDER=[(m,c) for m in list(METHODS)[:3] for c in ('bh','by')]+[(m,'native') for m in NATIVE]
STAT_LABELS=['Bac\nBH','Bac\nBY','Gene\nBH','Gene\nBY','Join\nBH','Join\nBY','SHOE',
             'MW\neBH','Hurdle\nproduct','Hurdle\naverage']
SHUFFLED_ORDER=[('bacteria','by'),('gene','by'),('join','by'),('exact_categorical','native')]
SHUFFLED_LABELS=['Bacteria BY','Gene BY','Join BY','Bins e-value']
DISPLAY_METHODS=dict(zip([m for m,_ in SHUFFLED_ORDER],SHUFFLED_LABELS))
COLORS['exact_categorical']='#CCBB44'
BINS_CV=CACHE/'cv/bins'


def _shuffle_count_summary(frame,reference):
    """Summarize fixed-data Monte Carlo counts, not feature-level FDR uncertainty."""
    values=frame.discoveries.to_numpy()
    n,r,m=reference.tested_features,reference.real_discoveries,reference.permutations
    if (not np.isfinite([n,r,m]).all() or any(x!=int(x) for x in (n,r,m)) or
            n<0 or not 0<=r<=n or m!=SHUFFLE_COUNT or len(frame)!=m or
            frame.permutation.duplicated().any() or set(frame.permutation)!=set(range(int(m))) or
            not frame.method.eq(reference.method).all() or not frame.correction.eq('by').all() or
            reference.correction!='by' or not np.isfinite(values).all() or
            (values<0).any() or (values>n).any() or (values!=np.floor(values)).any()):
        raise ValueError('Invalid or incomplete saved discovery counts')
    positive=values[values>0]; k=len(positive); b=int(np.count_nonzero(values>=r))
    p=(b+1)/(m+1)
    for field,expected in (('p_value',p),('exceedances',b),
                           ('mean_shuffled_discoveries',values.mean()),('log_p_value',np.log(p))):
        if field in reference and (not np.isfinite(reference[field]) or
                not np.isclose(reference[field],expected,rtol=0,atol=1e-14)):
            raise ValueError(f'Inconsistent saved {field}')
    if 'p_value' not in reference:
        raise ValueError('Missing saved permutation p-value')
    row=dict(dataset=reference.dataset,method=reference.method,correction='by',
        tested_features=int(n),real_discoveries=int(r),permutations=int(m),
        nonzero_shuffles=k,exceedances=b,
        positive_count_median=float(np.median(positive)) if k else np.nan,
        positive_count_max=int(positive.max()) if k else np.nan,
        p_value=float(reference.p_value),confidence_level=.95)
    for name,count,known in (('probability_positive',k,n==0),
                              ('probability_at_least_real',b,r==0)):
        estimate=count/m
        interval=(estimate,estimate) if known else binomtest(count,int(m)).proportion_ci(
            confidence_level=.95,method='exact')
        row.update({name:estimate,name+'_lower':interval[0],name+'_upper':interval[1]})
    return row


def shuffled_discovery_uncertainty():
    """Read complete saved BY counts; write one table and one descriptive overview."""
    protocol=json.loads((SHUFFLED/'protocol.json').read_text())
    if protocol['permutations']!=SHUFFLE_COUNT:
        raise ValueError('Saved shuffle budget differs from the configured budget')
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    rows=[]
    for dataset in DATASETS:
        for method,correction in SHUFFLED_ORDER[:3]:
            if correction!='by' or protocol['comparisons'].get(method)!=correction:
                raise ValueError('Expected saved BY baseline comparisons')
            directory=SHUFFLED/dataset/method
            if not complete(directory/'complete.json',signature):
                raise ValueError(f'Incomplete saved BY comparison: {directory}')
            reference=pd.read_csv(directory/'statistics.csv',float_precision='round_trip')
            if len(reference)!=1 or reference.iloc[0].dataset!=dataset or reference.iloc[0].method!=method:
                raise ValueError(f'Invalid saved comparison identity: {directory}')
            rows.append(_shuffle_count_summary(pd.read_csv(directory/'counts.csv'),reference.iloc[0]))
    table=pd.DataFrame(rows)
    fig,axes=plt.subplots(2,4,figsize=(24,11),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        part=table[table.dataset.eq(dataset)].set_index('method'); labels=[]
        for x,(method,_) in enumerate(SHUFFLED_ORDER[:3]):
            row=part.loc[method]
            for field,marker,offset in (('probability_positive','o',-.10),
                                         ('probability_at_least_real','D',.10)):
                value=row[field]
                ax.errorbar(x+offset,value,yerr=[[value-row[field+'_lower']],
                    [row[field+'_upper']-value]],fmt=marker,color=COLORS[method],
                    markersize=6,capsize=4,elinewidth=1.5)
            severity=(f'{row.positive_count_median:g}/{row.positive_count_max:g}'
                      if row.nonzero_shuffles else '—')
            labels.append(f'{DISPLAY_METHODS[method]}\nK={row.nonzero_shuffles}\nmedian/max={severity}')
        ax.set(title=dataset.replace('_unique',''),xticks=range(3),xticklabels=labels,
               ylabel='Probability',ylim=(0,1.08))
        ax.set_yscale('symlog',linthresh=.002)
        ax.set_yticks([0,.001,.01,.1,1],labels=['0','.001','.01','.1','1'])
        ax.tick_params(axis='x',labelsize=8); ax.grid(axis='y',alpha=.2)
    for ax in list(axes.flat)[len(DATASETS):]: ax.set_visible(False)
    fig.legend(handles=[Line2D([],[],color='black',marker='o',linestyle='none',
        label='Any discovery: P(D > 0)'),Line2D([],[],color='black',marker='D',linestyle='none',
        label='At least real count: P(D ≥ R)')],loc='lower center',ncol=2,bbox_to_anchor=(.5,.055))
    fig.suptitle(f'Shuffled discovery counts; {SHUFFLE_COUNT} shuffles\n'
                 'Probabilities with pointwise exact 95% Monte Carlo intervals')
    fig.text(.5,.02,'Intervals describe independent, uniform, with-replacement shuffle sampling, not participant or FDR uncertainty.\n'
             'K = positive shuffles; median/max use positive counts only. Probabilities are not permutation p-values. '
             'Incomplete Bins e-value omitted.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.13,1,.92))
    TABLES.mkdir(parents=True,exist_ok=True)
    table.to_csv(TABLES/'shuffled_discovery_count_summary.csv',index=False)
    fig.savefig(OUTPUT/'shuffled_discovery_count_uncertainty_overview.png',dpi=150)
    plt.close(fig)
    return table


def shuffled_discovery_distribution():
    """Plot saved complete shuffle counts only; never run or resume selectors."""
    protocol=json.loads((SHUFFLED/'protocol.json').read_text())
    if protocol['permutations']!=SHUFFLE_COUNT:
        raise ValueError('Saved shuffle budget differs from the configured budget')
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    fig,axes=plt.subplots(2,4,figsize=(24,10),sharey=True)
    handles={}; missing=set()
    for ax,dataset in zip(axes.flat,DATASETS):
        for (method,correction),label,marker in zip(SHUFFLED_ORDER,SHUFFLED_LABELS,['o','^','s','D']):
            directory=SHUFFLED/dataset/method
            if not complete(directory/'complete.json',signature):
                missing.add(label); continue
            frame=pd.read_csv(directory/'counts.csv')
            values=frame.discoveries.to_numpy()
            if (len(frame)!=SHUFFLE_COUNT or frame.permutation.duplicated().any() or
                    set(frame.permutation)!=set(range(SHUFFLE_COUNT)) or
                    not frame.method.eq(method).all() or not frame.correction.eq(correction).all() or
                    not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any()):
                raise ValueError(f'Invalid saved discovery counts: {directory}')
            support,counts=np.unique(values,return_counts=True)
            probability=counts/SHUFFLE_COUNT
            ax.vlines(support,.5/SHUFFLE_COUNT,probability,color=COLORS[method],alpha=.25)
            handles[label]=ax.scatter(support,probability,color=COLORS[method],marker=marker,
                                     s=35,alpha=.8,label=label)
        ax.set(title=dataset.replace('_unique',''),xlabel='Shuffled discoveries (dᵢ)',
               ylabel='Fraction of shuffles',ylim=(.5/SHUFFLE_COUNT,1.2))
        ax.set_xscale('symlog',linthresh=1); ax.set_yscale('log')
        ax.set_xlim(left=-.05)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(DATASETS):]: ax.set_visible(False)
    if not handles:
        plt.close(fig)
        raise ValueError('No complete saved shuffle distributions')
    fig.legend(handles.values(),handles.keys(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.035))
    if missing:
        fig.text(.5,.015,'Incomplete 999-shuffle results omitted: '+', '.join(sorted(missing)),ha='center')
    fig.suptitle(f'Shuffled discovery counts; {SHUFFLE_COUNT} shuffles\nEmpirical probability mass; full datasets')
    fig.tight_layout(rect=(0,.09,1,.92))
    path=OUTPUT/'shuffled_discovery_count_distribution_overview.png'
    fig.savefig(path,dpi=150); plt.close(fig)
    return path


def permutation_p_values():
    marker=SHUFFLED/'complete.json'
    if not marker.exists(): return None
    protocol=json.loads((SHUFFLED/'protocol.json').read_text())
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    complete(marker,signature)
    validate_fingerprints(dict(protocol['inputs'],**protocol['code_sha256']))
    for dataset in DATASETS:
        for group,_ in SHUFFLED_ORDER:
            if not complete(SHUFFLED/dataset/group/'complete.json',signature): raise ValueError('Incomplete shuffled test')
    table=pd.read_csv(SHUFFLED/'p_values.csv',float_precision='round_trip')
    expected={(d,m,c) for d in DATASETS for m,c in SHUFFLED_ORDER}
    if (table.duplicated(['dataset','method','correction']).any() or
            set(map(tuple,table[['dataset','method','correction']].to_numpy()))!=expected or
            not table.permutations.eq(SHUFFLE_COUNT).all() or not table.p_value_method.eq('permutation').all() or
            not table.p_value.between(1/(SHUFFLE_COUNT+1),1).all()):
        raise ValueError('Invalid completed permutation p-values')
    np.testing.assert_allclose(table.p_value,(table.exceedances+1)/(SHUFFLE_COUNT+1),rtol=0,atol=1e-15)
    np.testing.assert_allclose(table.log_p_value,np.log(table.p_value),rtol=0,atol=1e-14)
    return table


def publish_shuffled_p_values():
    table=permutation_p_values()
    if table is None: raise ValueError('The configured shuffle budget is not complete')
    comparison=pd.read_csv(SHUFFLED/'p_value_comparison.csv',float_precision='round_trip')
    keys=['dataset','method','correction']
    pd.testing.assert_frame_equal(table.sort_values(keys).reset_index(drop=True),
        comparison[table.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
    if not np.isfinite(comparison.normal_log_p_value).all() or comparison.normal_log_p_value.gt(0).any():
        raise ValueError('Invalid saved approximation log-p-values')
    table=table.merge(comparison[keys+['normal_log_p_value']],on=keys,validate='one_to_one')
    for previous,current in ((TABLES/'statistics_comparison.csv',table[table.method.isin(METHODS)]),
                             (CACHE/'evalue_power/preview/statistics.csv',table)):
        keys=['dataset','method','correction']; fields=keys+['samples','tested_features','real_discoveries']
        before=pd.read_csv(previous)[fields].sort_values(keys).reset_index(drop=True)
        after=current[fields].sort_values(keys).reset_index(drop=True)
        before=before.merge(after[keys],on=keys).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(before,after,check_dtype=False,check_exact=True)
    # Reuse the saved comparisons without recalculating p-values.
    statistical_figures(table,pd.DataFrame(),
        order=SHUFFLED_ORDER,labels=SHUFFLED_LABELS,
        colors=dict(COLORS,exact_categorical='#CCBB44'),fields=['negative_log10_p'])
    table.drop(columns='normal_log_p_value').to_csv(CACHE/'shuffled_fdp_test/p_values.csv',index=False)


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
            'mean_shuffled_discoveries','exceedances','p_value','log_p_value','p_value_method','status']
    frames[2]=frames[2][[c for c in fields if c in frames[2]]]
    for name,frame in zip(('prediction_comparison','prediction_fold_sd','statistics_comparison','pairwise_comparison'),frames):
        frame.to_csv(TABLES/f'{name}.csv',index=False)
    return frames


def prediction_figures(mean,sd):
    mean=mean[mean.method.isin(DISPLAY_METHODS)]
    sd=sd[sd.method.isin(DISPLAY_METHODS)]
    complete=[d for d in DATASETS if set(mean[mean.dataset.eq(d)].method)==set(DISPLAY_METHODS)]
    for suffix,ks in GRIDS.items():
        fig,axes=plt.subplots(2,4,figsize=(24,10),sharex=True,sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            for method,label in DISPLAY_METHODS.items():
                curve=mean[mean.dataset.eq(dataset)&mean.method.eq(method)&mean.requested_k.isin(ks)].sort_values('requested_k')
                if curve.empty: continue
                spread=sd[sd.dataset.eq(dataset)&sd.method.eq(method)].set_index('requested_k').reindex(curve.requested_k).auroc.to_numpy()
                ax.plot(curve.requested_k,curve.auroc,color=COLORS[method],label=label,marker='o',markersize=3)
                ax.fill_between(curve.requested_k,np.clip(curve.auroc-spread,0,1),np.clip(curve.auroc+spread,0,1),color=COLORS[method],alpha=.08)
            ax.set(title=dataset.replace('_unique',''),xlabel='K',ylabel='AUROC',xticks=ks,ylim=(0,1.01))
            if dataset not in complete: ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
            ax.grid(alpha=.2)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=4)
        fig.suptitle('AUROC vs K; 5-fold CV\nMean ± fold SD; unscreened Top-K')
        fig.tight_layout(rect=(0,.08,1,.93)); fig.savefig(OUTPUT/f'AUC_overview_k_{suffix}.png',dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        aggregate=mean[mean.dataset.isin(complete)].groupby(['method','requested_k']).auroc.mean().reset_index()
        for method,label in DISPLAY_METHODS.items():
            curve=aggregate[aggregate.method.eq(method)&aggregate.requested_k.isin(ks)].sort_values('requested_k')
            ax.plot(curve.requested_k,curve.auroc,color=COLORS[method],marker='o',label=label)
        ax.set(xlabel='K',ylabel='AUROC',xticks=ks,ylim=(0,1.01),
               title='Average AUROC vs K; 5-fold CV\nEqual dataset weights; unscreened Top-K')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(OUTPUT/f'AUC_average_k_{suffix}.png',dpi=150); plt.close(fig)


def statistical_figures(statistics,correlations,*,output=None,order=None,labels=None,colors=None,fields=None):
    output=OUTPUT if output is None else Path(output)
    order=STAT_ORDER if order is None else order
    labels=STAT_LABELS if labels is None else labels
    colors=COLORS if colors is None else colors
    for field,title in [('negative_log10_p','−log10(p-value)'),('real_discoveries','Discoveries'),
                        ('average_pairwise_absolute_correlation','Average pairwise |correlation|')]:
        if fields is not None and field not in fields: continue
        if field=='negative_log10_p' and ('p_value_method' not in statistics or
                not statistics.p_value_method.eq('permutation').all() or not statistics.permutations.eq(999).all()):
            raise ValueError('P-value plotting requires the completed 999-shuffle test')
        frame=correlations if field.startswith('average_pairwise') else statistics.assign(negative_log10_p=-statistics.log_p_value/np.log(10))
        frame=frame.merge(pd.DataFrame(order,columns=['method','correction']),on=['method','correction'],validate='many_to_one')
        fig,axes=plt.subplots(2,4,figsize=(24,10),sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            part=frame[frame.dataset.eq(dataset)].set_index(['method','correction'])
            for i,(method,correction) in enumerate(order):
                if (method,correction) not in part.index: continue
                row=part.loc[(method,correction)]
                overlay=field=='negative_log10_p' and 'normal_log_p_value' in frame
                if overlay:
                    ax.bar(i,-row.normal_log_p_value/np.log(10),width=.8,color=colors[method],alpha=.25,
                           edgecolor=colors[method],linewidth=1.5)
                if np.isfinite(row[field]): ax.bar(i,row[field],width=.45 if overlay else .8,
                    color=colors[method])
                elif field.startswith('average_pairwise'):
                    ax.text(i,.95,'skipped' if row.status=='skipped_size_limit' else 'undefined',rotation=90,
                            va='top',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
            if len(part)!=len(order): ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
            ax.set(title=dataset.replace('_unique',''),xticks=range(len(order)),xticklabels=labels,
                   xlim=(-.6,len(order)-.4),ylabel=title)
            ax.tick_params(axis='x',labelsize=7); ax.grid(axis='y',alpha=.2)
            if field=='negative_log10_p':
                if 'normal_log_p_value' in frame:
                    upper=max(3.15,-min(frame.log_p_value.min(),frame.normal_log_p_value.min())/np.log(10)*1.15)
                    ax.set_yscale('symlog',linthresh=3); ax.set_ylim(0,upper)
                else: ax.set_ylim(0,3.15)
            elif field=='real_discoveries': ax.set_yscale('symlog',linthresh=1)
            else: ax.set_ylim(0,1.05)
            if field=='negative_log10_p': ax.axhline(-np.log10(.05),color='black',linestyle=':')
        if field=='negative_log10_p':
            if 'normal_log_p_value' in frame:
                fig.legend(handles=[Patch(facecolor='gray',label='Permutation p-value (solid narrow bar, foreground)'),
                    Patch(facecolor='gray',alpha=.25,label='Binomial/Normal p-value (transparent wide bar, background)')],
                    loc='lower center',ncol=2)
        fig.suptitle(f'-log10(o-value); {SHUFFLE_COUNT} shuffles\nPermutation vs Binomial/Normal approximation'
                     if field=='negative_log10_p' else
                     'Discoveries; full datasets' if field=='real_discoveries' else
                     'Average pairwise |correlation|; full datasets\nComplete discovery sets')
        fig.tight_layout(rect=(0,.05 if field=='negative_log10_p' else 0,1,.92)); fig.savefig(output/f'{field}_overview.png',dpi=150); plt.close(fig)


def evalue_power_preview():
    """Temporary discoveries/correlation views; p-values have a separate four-method view."""
    root=CACHE/'evalue_power/bins'; output=CACHE/'evalue_power/preview'
    statistics=pd.read_csv(TABLES/'statistics_comparison.csv',float_precision='round_trip')
    correlations=pd.read_csv(TABLES/'pairwise_comparison.csv',float_precision='round_trip')
    new_correlations=[]
    for dataset in DATASETS:
        directory=root/dataset; marker=directory/'complete.json'
        state=json.loads(marker.read_text()); complete(marker,state['signature'])
        identity=state['input_identity']
        if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=state['signature']:
            raise ValueError(f'Changed search identity: {dataset}')
        expected=dict(identity['inputs'])
        expected.update(identity['protocol']['code'])
        expected.update(identity['protocol']['protected_main_scientific_code'])
        validate_fingerprints(expected)
        reference=pd.read_csv(directory/'comparison.csv').iloc[0]
        with np.load(directory/'evidence.npz') as saved:
            ids=saved['feature_ids']; discoveries=saved['discovery_ids']
        columns=np.flatnonzero(np.isin(ids,discoveries)); r=len(discoveries)
        if len(columns)!=r or r!=int(reference.discoveries):
            raise ValueError(f'Invalid discovery membership: {dataset}')
        metadata=dict(dataset=dataset,method='exact_categorical',correction='native')
        statistics=pd.concat([statistics,pd.DataFrame([dict(metadata,samples=int(reference.samples),
            real_discoveries=r,tested_features=int(reference.full_family))])],ignore_index=True)
        # The helper applies its size guard before allocating dense values/correlations.
        matrix=sparse.load_npz(NATIVE_FULL/dataset/'raw_pairs/features.npz') if 2<=r<=PAIRWISE_FEATURE_LIMIT else None
        if matrix is not None and matrix.shape!=(int(reference.samples),len(ids)):
            raise ValueError(f'Misaligned raw predictors: {dataset}')
        new_correlations.append(dict(metadata,selection='all_significant',requested_k=np.nan,
            **average_pairwise_absolute_correlation(matrix,columns)))
        del matrix
        print(f'{dataset}: discoveries={r}; correlation={new_correlations[-1]["status"]}',flush=True)
    correlations=pd.concat([correlations,pd.DataFrame(new_correlations)],ignore_index=True)
    order=STAT_ORDER+[('exact_categorical','native')]
    for frame in (statistics,correlations):
        if frame.duplicated(['dataset','method','correction']).any() or set(map(tuple,
                frame[['dataset','method','correction']].to_numpy()))!={(d,m,c) for d in DATASETS for m,c in order}:
            raise ValueError('Incomplete/duplicate preview table')
    output.mkdir(parents=True,exist_ok=True)
    statistics.to_csv(output/'statistics.csv',index=False)
    correlations.to_csv(output/'pairwise_correlations.csv',index=False)
    statistical_figures(statistics,correlations,output=output,order=order,
        labels=STAT_LABELS+['Bins\ne-value'],colors=dict(COLORS,exact_categorical='#CCBB44'),
        fields=['real_discoveries','average_pairwise_absolute_correlation'])
    print(f'Preview complete: {output}',flush=True)


def ranked_pairwise_correlations():
    groups=[]
    for dataset in DATASETS:
        if (FULL/dataset/'complete.json').exists():
            groups.extend((dataset,FULL/dataset/f'{m}_features.npz',[(m,None)]) for m in list(METHODS)[:3])
        directory=NATIVE_FULL/dataset
        if (directory/'complete.json').exists():
            groups.append((dataset,directory/'raw_pairs/features.npz',
                           [('exact_categorical',CACHE/'evalue_power/bins'/dataset/'evidence.npz')]))
    sources=sorted({p for _,matrix,methods in groups for p in [matrix,*[rank for _,rank in methods if rank is not None]]})
    identity=dict(version='four_methods_by_bins_v1',ranking='MW p/ID; Bins native log-evidence/ID',
        metric=file_hash(PROJECT/'metrics/correlation.py'),
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
                with np.load(ranking) as saved:
                    order=np.lexsort((saved['feature_ids'],-saved['log_e']))
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
        for method,label in DISPLAY_METHODS.items():
            curve=part[part.method.eq(method)].sort_values('requested_k')
            ax.plot(curve.requested_k,curve.average_pairwise_absolute_correlation,
                    color=COLORS[method],label=label,marker='o',markersize=3)
        if len(part)!=len(DISPLAY_METHODS)*len(ks): ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
        ax.set(title=dataset.replace('_unique',''),xlabel='K',xticks=ks,ylim=(0,1.01),
               ylabel='Average pairwise |correlation|')
        ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=4)
    fig.suptitle('Average pairwise |correlation| vs K\nUnscreened Top-K; full datasets')
    fig.tight_layout(rect=(0,.08,1,.93))
    fig.savefig(OUTPUT/'average_pairwise_absolute_correlation_k_10_20_50_75.png',dpi=150); plt.close(fig)


def main():
    mean,sd,statistics,correlations=collect_tables()
    extra_mean=[];extra_sd=[]
    for dataset in DATASETS:
        for fold in range(5):
            directory=BINS_CV/dataset/f'fold_{fold}'
            protocol=json.loads((directory/'protocol.json').read_text())
            signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            if not complete(directory/'complete.json',signature):raise ValueError('Incomplete Bins CV')
            validate_fingerprints(dict(protocol['inputs'],**protocol['code']))
        extra_mean.append(pd.read_csv(BINS_CV/dataset/'mean_fold_metrics.csv',float_precision='round_trip').assign(dataset=dataset))
        extra_sd.append(pd.read_csv(BINS_CV/dataset/'sd_fold_metrics.csv',float_precision='round_trip').assign(dataset=dataset))
    mean=pd.concat([mean,*extra_mean],ignore_index=True);sd=pd.concat([sd,*extra_sd],ignore_index=True)
    statistics=permutation_p_values()
    if statistics is None:raise ValueError('Missing four-method discovery counts')
    correlations=pd.read_csv(CACHE/'evalue_power/preview/pairwise_correlations.csv',float_precision='round_trip')
    prediction_figures(mean,sd)
    statistical_figures(statistics,correlations,order=SHUFFLED_ORDER,labels=SHUFFLED_LABELS,
        fields=['real_discoveries','average_pairwise_absolute_correlation'])
    publish_shuffled_p_values()
    ranked_correlation_figure()


if __name__=='__main__': main()
