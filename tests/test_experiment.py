"""Small scientific fixtures and strict cache/relocation checks; no real refits."""
import itertools
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.special import logsumexp
from methods.selectors import shoe, evalue as selectors, mann_whitney as mw
from methods import preprocessing as prep
from methods.representations import collapsed_average, aggregate_pair_clusters
from methods.graphs.worker import (MAX_COMMUNITY_EDGES,choose_effective_threshold,count_threshold_edges,
    fallback_thresholds,pack_sparse_detection,stable_cluster_labels)
from experiments import cross_validation as study, full_dataset as runner, cache
from experiments.cache import JOIN_PYTHON, PROJECT
from metrics import correlation as u
from metrics.prediction import evaluate, predict_model, new_model
from shuffled_fdp_test.statistic import (discovery_statistics, label_permutations,
                                        normal_approximation_statistics, permutation_discoveries)


class PipelineTests(unittest.TestCase):
    def test_power_preview_is_isolated_and_guards_large_discovery_sets(self):
        import hashlib
        from contextlib import ExitStack
        from scripts import plot_comparison as plotting
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); table=root/'tables'; directory=root/'evalue_power/bins/toy'
            raw=root/'native/toy/raw_pairs'; table.mkdir(); directory.mkdir(parents=True);raw.mkdir(parents=True)
            original_output=plotting.OUTPUT
            for name,value in dict(CACHE=root,TABLES=table,NATIVE_FULL=root/'native',DATASETS=['toy']).items():
                stack.enter_context(patch.object(plotting,name,value))
            base=[dict(dataset='toy',method=m,correction=c) for m,c in plotting.STAT_ORDER]
            pd.DataFrame([dict(row,real_discoveries=0,log_p_value=0.) for row in base]).to_csv(table/'statistics_comparison.csv',index=False)
            pd.DataFrame([dict(row,**u.average_pairwise_absolute_correlation(None,[])) for row in base]).to_csv(table/'pairwise_comparison.csv',index=False)
            baseline=(table/'statistics_comparison.csv').read_bytes()
            x=sparse.csr_matrix([[1,2,0],[2,1,1],[3,2,4],[4,1,9],[5,2,16],[6,1,25]])
            sparse.save_npz(raw/'features.npz',x)
            identity=dict(inputs={},protocol=dict(code={},protected_main_scientific_code={}))
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            for r in (2,10001,0):
                ids=np.arange(max(3,r))+10; discoveries=ids[[0,2]] if r==2 else ids[:r]
                np.savez_compressed(directory/'evidence.npz',feature_ids=ids,discovery_ids=discoveries)
                pd.DataFrame([dict(samples=6,full_family=20000,discoveries=r)]).to_csv(directory/'comparison.csv',index=False)
                marker=directory/'complete.json'
                cache.seal(marker,key,[directory/n for n in ('evidence.npz','comparison.csv')])
                cache.write_json(marker,dict(json.loads(marker.read_text()),input_identity=identity))
                active=pd.DataFrame([
                    dict(row,samples=6,**discovery_statistics([0]*999,0,20000)) for row in base]+[
                    dict(dataset='toy',method='exact_categorical',correction='native',samples=6,
                         **discovery_statistics([0]*999,r,20000))])
                preview=root/'evalue_power/preview';preview.mkdir(exist_ok=True)
                active.to_csv(preview/'statistics.csv',index=False)
                with patch.object(plotting,'statistical_figures') as figures:
                    if r==2: plotting.evalue_power_preview()
                    else:
                        with patch.object(sparse,'load_npz',side_effect=AssertionError('Loaded guarded matrix')):plotting.evalue_power_preview()
                statistics,correlations=figures.call_args.args
                new=statistics[statistics.method.eq('exact_categorical')].iloc[0]
                self.assertEqual(new.real_discoveries,r);self.assertEqual(new.tested_features,20000)
                actual=correlations[correlations.method.eq('exact_categorical')].iloc[0]
                expected=u.average_pairwise_absolute_correlation(x,[0,2]) if r==2 else u.average_pairwise_absolute_correlation(None,range(r))
                self.assertEqual(actual.status,expected['status'])
                if r==2:self.assertAlmostEqual(actual.average_pairwise_absolute_correlation,expected['average_pairwise_absolute_correlation'])
                self.assertEqual(figures.call_args.kwargs['output'],root/'evalue_power/preview')
                self.assertNotIn('negative_log10_p',figures.call_args.kwargs['fields'])
                self.assertEqual(plotting.OUTPUT,original_output)
                self.assertEqual(baseline,(table/'statistics_comparison.csv').read_bytes())

    def test_exact_categorical_evidence_and_fixed_family(self):
        from scripts import search_evalue_power as trial
        x=sparse.csr_matrix([[0,1,0],[0,1,1],[1,1,2],[2,1,3],[4,1,4],[8,1,5.]])
        labels=np.array([np.isin(np.arange(6),s).astype(int) for s in itertools.combinations(range(6),3)])
        prepared=trial.prepare_bins(x)
        log_e=trial.evidence(prepared,labels)
        np.testing.assert_allclose(np.exp(log_e).mean(axis=0),1,atol=1e-12)
        np.testing.assert_array_equal(trial.evidence(prepared,labels),log_e)
        counts=prepared[2][0];mass,null=trial.table(*counts,3)
        score=trial.log_lr(counts,np.array([0,1,2]),3)
        expected=np.exp(score)/np.sum(np.exp(mass+null))
        self.assertAlmostEqual(np.exp(trial.evidence(prepared,[[0,0,0,1,1,1]])[0,0]),expected)
        np.testing.assert_allclose(log_e[:,1],0,atol=1e-12)
        np.testing.assert_array_equal(trial.select(np.log([30.,30.,1.]),[2,0,1],3),[1,0])
        self.assertEqual(len(trial.select(np.log([100.]),[0],10000)),0)
        self.assertEqual(len(trial.select([],[],0)),0)
        self.assertEqual(trial.evidence(trial.prepare_bins(x[:,:0]),labels).shape,(20,0))
        with self.assertRaises(ValueError):trial.evidence(trial.prepare_bins(x),np.ones((1,6)))
        with self.assertRaises(ValueError):trial.select([1.],[3],3)

    def test_isolated_power_search_resume_and_input_rejection(self):
        from contextlib import ExitStack
        from scripts import search_evalue_power as trial
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); source=root/'inputs'; native=root/'native'
            output=root/'bins'; case=native/'toy/raw_pairs'
            for path in (source/'toy',case):path.mkdir(parents=True)
            y=np.tile([0,1],10)
            samples=pd.DataFrame(dict(sample_id=[f's{i}' for i in range(20)],subject_id=[f'p{i}' for i in range(20)],y=y))
            samples.to_csv(source/'toy/samples.csv',index=False)
            x=sparse.csr_matrix(np.column_stack((y+1,np.arange(20)%3)))
            sparse.save_npz(case/'features.npz',x)
            np.savez_compressed(case/'preprocessing.npz',feature_ids=[0,1],fit_indices=np.arange(20),prevalence_cutoff=5)
            cache.seal(case/'preprocessing_complete.json','fixture',[case/'features.npz',case/'preprocessing.npz'])
            np.savez_compressed(source/'toy/edges.npz',pair_gene=[0,1,2])
            for name,value in dict(ROOT=output,SOURCE=source,NATIVE_FULL=native,DATASETS=['toy']).items():
                stack.enter_context(patch.object(trial,name,value))
            stack.enter_context(patch.object(trial,'exact_checks',return_value={'fixture':True}))
            # Fixture inputs live outside PROJECT; the identity still records their actual paths.
            stack.enter_context(patch.object(trial,'PROJECT',root))
            stack.enter_context(patch.object(cache,'PROJECT',root))
            stack.enter_context(patch.object(cache,'PROVENANCE',root/'cache/provenance'))
            (root/'scripts').mkdir();(root/'experiments').mkdir()
            (root/'experiments/cache.py').write_text('fixture')
            cache.write_json(root/'cache/provenance/compatibility.json',dict(
                scientific_code_sha256={'experiments/cache.py':cache.file_hash(root/'experiments/cache.py')}))
            stack.enter_context(patch.object(trial,'CACHE',root/'cache'))
            (root/'scripts/search_evalue_power.py').write_bytes(Path(trial.__file__).read_bytes())
            stack.enter_context(patch.object(trial,'__file__',str(root/'scripts/search_evalue_power.py')))
            trial.run()
            first=(output/'toy/evidence.npz').read_bytes()
            with patch.object(trial,'prepare_bins',side_effect=AssertionError('Recomputed completed search')):trial.run()
            self.assertEqual(first,(output/'toy/evidence.npz').read_bytes())
            self.assertFalse((output/'toy/shuffled_counts.csv').exists())
            sparse.save_npz(case/'features.npz',x*2)
            with self.assertRaisesRegex(ValueError,'Changed scientific protocol'):trial.run()

    def test_ranked_correlation_uses_unscreened_scores_and_reuses_derived_cache(self):
        from contextlib import ExitStack
        from scripts import plot_comparison as plotting
        x=sparse.csr_matrix(np.random.default_rng(42).uniform(size=(20,80)))
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); base=root/'full/toy'; native=root/'native/toy'
            base.mkdir(parents=True); native.mkdir(parents=True)
            for directory in (base,native): (directory/'complete.json').write_text('{}')
            for method in list(cache.METHODS)[:3]: sparse.save_npz(base/f'{method}_features.npz',x)
            for family in ('shoe','raw_pairs'):
                (native/family).mkdir(); sparse.save_npz(native/family/'features.npz',x)
            evidence=root/'evalue_power/bins/toy/evidence.npz';evidence.parent.mkdir(parents=True)
            np.savez_compressed(evidence,feature_ids=np.arange(80),log_e=np.arange(80))
            for method in cache.NATIVE:
                case=native/'shoe' if method=='shoe' else native/'raw_pairs'/method
                case.mkdir(exist_ok=True)
                np.savez_compressed(case/'selection.npz',order=np.arange(79,-1,-1),native=np.array([],int))
            for name,value in dict(CACHE=root,FULL=root/'full',NATIVE_FULL=root/'native',DATASETS=['toy'],TABLES=root/'tables').items():
                stack.enter_context(patch.object(plotting,name,value))
            actual=plotting.ranked_pairwise_correlations()
            self.assertEqual(len(actual),4*4)
            self.assertEqual(set(actual.method),set(plotting.DISPLAY_METHODS))
            self.assertTrue(actual.effective_k.eq(actual.requested_k).all())
            for row in actual.itertuples():
                order=np.arange(79,-1,-1) if row.method=='exact_categorical' else np.arange(80)
                expected=u.average_pairwise_absolute_correlation(x,order[:row.requested_k])
                self.assertAlmostEqual(row.average_pairwise_absolute_correlation,expected['average_pairwise_absolute_correlation'])
            with patch.object(sparse,'load_npz',side_effect=AssertionError('Derived cache recomputed')):
                pd.testing.assert_frame_equal(actual,plotting.ranked_pairwise_correlations())
            np.savez_compressed(evidence,feature_ids=np.arange(80),log_e=-np.arange(80))
            changed=plotting.ranked_pairwise_correlations()
            rows=changed[changed.method.eq('exact_categorical')].set_index('requested_k')
            self.assertAlmostEqual(rows.loc[10,'average_pairwise_absolute_correlation'],u.average_pairwise_absolute_correlation(x,range(10))['average_pairwise_absolute_correlation'])

    def test_bins_cv_training_only_scores_ties_and_empty_family(self):
        from scripts import evaluate_bins_cv as cv
        rng=np.random.default_rng(42)
        x=sparse.csr_matrix(rng.integers(0,4,size=(20,5)))
        y=np.tile([0,1],10);a=np.arange(16);ids=np.array([4,1,9,2,5])
        result=cv.training_selection(x,ids,a,y,10)
        changed=x.toarray();changed[16:]=10000
        shuffled=y.copy();shuffled[16:]=1-shuffled[16:]
        other=cv.training_selection(sparse.csr_matrix(changed),ids,a,shuffled,10)
        for key in result:np.testing.assert_array_equal(result[key],other[key])
        np.testing.assert_array_equal(result['order'],np.lexsort((ids,-result['log_e'])))
        empty=cv.training_selection(x[:,:0],[],a,y,10)
        self.assertEqual(len(empty['order']),0)
        with self.assertRaises(ValueError):cv.training_selection(sparse.csr_matrix(np.ones((20,1))),[0],a,y,10)

    def test_bins_cv_cached_folds_resume_summaries_and_protocol_rejection(self):
        from contextlib import ExitStack
        from scripts import evaluate_bins_cv as cv
        rng=np.random.default_rng(42);y=np.tile([0,1],10)
        x=sparse.csr_matrix(rng.uniform(.1,2,size=(20,4)))
        samples=pd.DataFrame(dict(sample_id=[f's{i}' for i in range(20)],y=y))
        with tempfile.TemporaryDirectory(dir=PROJECT) as temporary,ExitStack() as stack:
            root=Path(temporary);source=root/'inputs';base=root/'base';native=root/'native';output=root/'bins'
            (source/'toy').mkdir(parents=True)
            samples.to_csv(source/'toy/samples.csv',index=False)
            np.savez_compressed(source/'toy/edges.npz',pair_gene=np.arange(4))
            for name,value in dict(SOURCE=source,CV=base,NATIVE_CV=native,OUTPUT=output).items():
                stack.enter_context(patch.object(cv,name,value))
            stack.enter_context(patch.object(study,'OUTPUT',base))
            stack.enter_context(patch.object(study,'cluster',side_effect=AssertionError('Graph construction')))
            stack.enter_context(patch.object(mw,'mann_whitney_many',side_effect=AssertionError('MW rerun')))
            folds=study.make_folds(y)
            for fold,(a,b) in enumerate(folds):
                path=native/'toy'/f'fold_{fold}';raw=path/'raw_pairs';raw.mkdir(parents=True)
                baseline=base/'toy'/f'fold_{fold}';baseline.mkdir(parents=True)
                sparse.save_npz(raw/'features.npz',x)
                np.savez_compressed(raw/'preprocessing.npz',feature_ids=np.arange(4),fit_indices=a,
                    evidence_indices=np.array([],int),prevalence_cutoff=5)
                np.savez_compressed(path/'outer_split.npz',train_index=a,test_index=b)
                np.savez_compressed(baseline/'split.npz',train_index=a,test_index=b)
                cache.seal(raw/'preprocessing_complete.json','fixture',[raw/'features.npz',raw/'preprocessing.npz'])
                cache.seal(path/'complete.json','fixture',[raw/'features.npz',raw/'preprocessing.npz',path/'outer_split.npz'])
                cv.fold_run('toy',fold,samples,a,b,4,'fixture')
            study.summarize_dataset('toy',samples,output,[cv.METHOD])
            mean=pd.read_csv(output/'toy/mean_fold_metrics.csv')
            self.assertTrue(mean.n_folds.eq(5).all());self.assertEqual(len(mean),len(cache.K_VALUES))
            for fold,(a,b) in enumerate(folds):
                with patch.object(cv,'training_selection',side_effect=AssertionError('Selection rerun')),\
                        patch.object(cv,'evaluate_selection',side_effect=AssertionError('Model rerun')):
                    cv.fold_run('toy',fold,samples,a,b,4,'fixture')
            with patch.object(cv,'K_VALUES',[1]):
                with self.assertRaisesRegex(ValueError,'Changed Bins CV checkpoint'):
                    cv.fold_run('toy',0,samples,*folds[0],4,'fixture')

    def test_public_prediction_figures_show_only_four_consistent_methods(self):
        from scripts import plot_comparison as plotting
        rows=[dict(dataset='toy',method=m,requested_k=k,auroc=.6)
            for m in [*plotting.DISPLAY_METHODS,'shoe'] for k in cache.K_VALUES]
        mean=pd.DataFrame(rows);sd=mean.assign(auroc=.05)
        def inspect(fig,*args,**kwargs):
            legend=fig.legends[0] if fig.legends else fig.axes[0].get_legend()
            self.assertEqual([t.get_text() for t in legend.get_texts()],list(plotting.DISPLAY_METHODS.values()))
            self.assertEqual(len(fig.axes[0].lines),4)
            self.assertEqual([line.get_color() for line in fig.axes[0].lines],
                [plotting.COLORS[m] for m in plotting.DISPLAY_METHODS])
        with tempfile.TemporaryDirectory() as temporary,patch.object(plotting,'DATASETS',['toy']),\
                patch.object(plotting,'OUTPUT',Path(temporary)),\
                patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=inspect) as figures:
            plotting.prediction_figures(mean,sd)
        self.assertEqual(figures.call_count,4)

    def test_shuffle_distribution_uses_all_saved_counts_and_rejects_incomplete_coverage(self):
        import hashlib
        from scripts import plot_comparison as plotting
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); protocol=dict(permutations=5)
            (root/'protocol.json').write_text(json.dumps(protocol))
            signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            directory=root/'toy/bacteria'; directory.mkdir(parents=True)
            path=directory/'counts.csv'
            frame=pd.DataFrame(dict(permutation=range(5),method='bacteria',correction='by',discoveries=[0,0,0,2,100]))
            frame.to_csv(path,index=False); cache.seal(directory/'complete.json',signature,[path])
            def inspect(fig,path,**kwargs):
                points=fig.axes[0].collections[1].get_offsets()
                np.testing.assert_allclose(points,[[0,.6],[2,.2],[100,.2]])
                self.assertIn('Bins e-value',fig.texts[0].get_text())
                self.assertEqual(fig.axes[0].get_xscale(),'symlog')
                self.assertEqual(fig.axes[0].get_yscale(),'log')
            with patch.object(plotting,'DATASETS',['toy']),patch.object(plotting,'SHUFFLED',root),\
                    patch.object(plotting,'OUTPUT',root),patch.object(plotting,'SHUFFLE_COUNT',5),\
                    patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=inspect):
                plotting.shuffled_discovery_distribution()
                frame.loc[4,'permutation']=0
                frame.to_csv(path,index=False); cache.seal(directory/'complete.json',signature,[path])
                with self.assertRaisesRegex(ValueError,'Invalid saved discovery counts'):
                    plotting.shuffled_discovery_distribution()
            plotting.plt.close('all')

    def test_shuffle_count_summary_boundaries_ties_and_exact_intervals(self):
        from scripts import plot_comparison as plotting
        def fixture(values,r,n):
            frame=pd.DataFrame(dict(permutation=range(999),method='join',correction='by',discoveries=values))
            reference=pd.Series(dict(dataset='toy',method='join',correction='by',
                **discovery_statistics(values,r,n)))
            return frame,reference
        frame,reference=fixture([0]*995+[1,1,1,716],395,143097)
        row=plotting._shuffle_count_summary(frame,reference)
        self.assertEqual((row['nonzero_shuffles'],row['exceedances']),(4,1))
        self.assertEqual((row['positive_count_median'],row['positive_count_max']),(1,716))
        self.assertEqual(row['p_value'],.002)
        expected=stats.binomtest(4,999).proportion_ci(.95,method='exact')
        self.assertEqual((row['probability_positive_lower'],row['probability_positive_upper']),tuple(expected))
        tied,tied_reference=fixture([0]*997+[5,5],5,10)
        self.assertEqual(plotting._shuffle_count_summary(tied,tied_reference)['exceedances'],2)
        zero,zero_reference=fixture([0]*999,1,10)
        result=plotting._shuffle_count_summary(zero,zero_reference)
        self.assertEqual(result['p_value'],.001)
        self.assertTrue(np.isnan(result['positive_count_median']))
        self.assertAlmostEqual(result['probability_positive_upper'],1-.025**(1/999))
        for n in (0,10):
            empty,empty_reference=fixture([0]*999,0,n)
            result=plotting._shuffle_count_summary(empty,empty_reference)
            self.assertEqual((result['probability_at_least_real_lower'],result['probability_at_least_real_upper']),(1,1))
            if n==0:self.assertEqual((result['probability_positive_lower'],result['probability_positive_upper']),(0,0))
        invalid=[frame.iloc[:-1],frame.assign(permutation=0),frame.assign(correction='bh'),
                 frame.assign(discoveries=-1),frame.assign(discoveries=143098),
                 frame.assign(discoveries=.5),frame.assign(discoveries=np.nan)]
        for changed in invalid:
            with self.subTest(invalid=changed.iloc[0].to_dict()),self.assertRaises(ValueError):
                plotting._shuffle_count_summary(changed,reference)
        for key,value in [('real_discoveries',-1),('tested_features',.5),('permutations',100),
                          ('p_value',.001),('exceedances',2),('mean_shuffled_discoveries',0)]:
            changed=reference.copy(); changed[key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):
                plotting._shuffle_count_summary(frame,changed)

    def test_shuffle_uncertainty_reads_saved_protocol_without_scientific_fits(self):
        import hashlib
        from contextlib import ExitStack
        from scripts import plot_comparison as plotting
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); protocol=dict(permutations=999,
                comparisons=dict(plotting.SHUFFLED_ORDER),code_sha256={'old-scientific-code':'historical'})
            (root/'protocol.json').write_text(json.dumps(protocol))
            signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            originals={}
            for method,_ in plotting.SHUFFLED_ORDER[:3]:
                directory=root/'toy'/method; directory.mkdir(parents=True)
                frame=pd.DataFrame(dict(permutation=range(999),method=method,correction='by',discoveries=[0]*999))
                frame.to_csv(directory/'counts.csv',index=False)
                pd.DataFrame([dict(dataset='toy',method=method,correction='by',
                    **discovery_statistics(frame.discoveries,1,10))]).to_csv(directory/'statistics.csv',index=False)
                cache.seal(directory/'complete.json',signature,[directory/'counts.csv',directory/'statistics.csv'])
            originals={path:path.read_bytes() for path in root.rglob('*') if path.is_file()}
            for name,value in dict(SHUFFLED=root,OUTPUT=root,TABLES=root/'tables',DATASETS=['toy']).items():
                stack.enter_context(patch.object(plotting,name,value))
            for module,name in [(runner,'run_shuffled_test'),(runner,'shuffled_protocol'),
                                (runner,'shuffled_counts'),(plotting,'validate_fingerprints')]:
                stack.enter_context(patch.object(module,name,side_effect=AssertionError('Scientific computation called')))
            def inspect(fig,path,**kwargs):
                self.assertEqual(Path(path).name,'shuffled_discovery_count_uncertainty_overview.png')
                self.assertEqual(len(fig.axes),8)
                self.assertEqual(fig.axes[0].get_yscale(),'symlog')
                self.assertIn('Monte Carlo',fig.texts[0].get_text())
                self.assertIn('Incomplete Bins e-value omitted',fig.texts[1].get_text())
            with patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=inspect):
                result=plotting.shuffled_discovery_uncertainty()
            saved=pd.read_csv(root/'tables/shuffled_discovery_count_summary.csv',float_precision='round_trip')
            pd.testing.assert_frame_equal(saved,result,check_exact=True)
            self.assertEqual(set(result.method),{'bacteria','gene','join'})
            for path,content in originals.items():self.assertEqual(path.read_bytes(),content)
            (root/'toy/join/complete.json').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete saved BY comparison'):
                plotting.shuffled_discovery_uncertainty()

    def test_public_discovery_correlation_bars_exclude_bh_and_other_methods(self):
        from scripts import plot_comparison as plotting
        rows=[dict(dataset='toy',method=m,correction=c,real_discoveries=i+1,log_p_value=0.,
            average_pairwise_absolute_correlation=(i+1)/10,status='defined')
            for i,(m,c) in enumerate([*plotting.SHUFFLED_ORDER,('bacteria','bh'),('shoe','native')])]
        frame=pd.DataFrame(rows)
        def inspect(fig,path,**kwargs):
            ax=fig.axes[0]
            self.assertEqual([t.get_text() for t in ax.get_xticklabels()],plotting.SHUFFLED_LABELS)
            self.assertEqual(len(ax.patches),4)
            scale=1 if 'real_discoveries' in str(path) else .1
            np.testing.assert_allclose([p.get_height() for p in ax.patches],np.arange(1,5)*scale)
            self.assertTrue(all(p.get_hatch() is None for p in ax.patches))
        with tempfile.TemporaryDirectory() as temporary,patch.object(plotting,'DATASETS',['toy']),\
                patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=inspect):
            plotting.statistical_figures(frame,frame,output=Path(temporary),
                order=plotting.SHUFFLED_ORDER,labels=plotting.SHUFFLED_LABELS,
                fields=['real_discoveries','average_pairwise_absolute_correlation'])

    def test_seven_method_small_run_and_completed_resume(self):
        from contextlib import ExitStack
        from experiments import full_dataset as full
        from scripts import plot_comparison as plotting
        rng=np.random.default_rng(42)
        y=np.tile([0,1],30)
        x=sparse.csr_matrix(np.column_stack((1+y,rng.integers(0,4,60),rng.integers(0,5,60),rng.integers(0,3,60))))
        samples=pd.DataFrame(dict(sample_id=[f's{i}' for i in range(60)],subject_id=[f'p{i}' for i in range(60)],y=y))
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); source=root/'inputs'; cv=root/'cv'; fullroot=root/'full'; nc=root/'native_cv'; nf=root/'native_full'
            (source/'toy').mkdir(parents=True)
            samples.to_csv(source/'toy/samples.csv',index=False); sparse.save_npz(source/'toy/X_valid.npz',x)
            np.savez_compressed(source/'toy/edges.npz',pair_bacteria=np.array([0,0,1,1]),pair_gene=np.array([0,1,0,1]))
            for module,values in [(study,dict(SOURCE=source,OUTPUT=cv,NATIVE_CV=nc,PROVENANCE=root/'provenance',DATASETS=['toy'])),
                                  (full,dict(SOURCE=source,OUTPUT=fullroot,NATIVE_FULL=nf,PROVENANCE=root/'provenance',DATASETS=['toy'],SHUFFLE_COUNT=9))]:
                for name,value in values.items(): stack.enter_context(patch.object(module,name,value))
                stack.enter_context(patch.object(module,'open_protocol',side_effect=lambda directory,stage:stage))
                stack.enter_context(patch.object(module,'cluster',return_value=(np.array([0,0,1,2]),dict(
                    clusters=3,effective_threshold=.7,join=dict(edge_count=1),community=dict(deduplicated_edges=1),threshold_edge_counts={'.7':1}))))
            stack.enter_context(patch.object(full,'label_permutations',side_effect=AssertionError('Selection must not shuffle labels')))
            stack.enter_context(patch.object(full,'permutation_discoveries',side_effect=AssertionError('Selection must not repeat shuffled tests')))
            study.main(); full.main()
            self.assertFalse(list(fullroot.rglob('*shuffled*')))
            self.assertFalse(list(nf.rglob('*shuffled*')))
            for module in (study,full):
                stack.enter_context(patch.object(module,'cluster',side_effect=AssertionError('Graph recomputed')))
            stack.enter_context(patch.object(study,'evaluate',side_effect=AssertionError('Prediction refit')))
            stack.enter_context(patch.object(full,'prepared',side_effect=AssertionError('Selection refit')))
            stack.enter_context(patch.object(full,'permutation_discoveries',side_effect=AssertionError('Permutations rerun')))
            study.main(); full.main()
            for name,value in dict(CV=cv,FULL=fullroot,NATIVE_CV=nc,NATIVE_FULL=nf,DATASETS=['toy'],TABLES=root/'tables',CACHE=root).items():
                stack.enter_context(patch.object(plotting,name,value))
            frames=plotting.collect_tables()
            self.assertEqual(len(frames[0]),7*len(cache.K_VALUES))
            self.assertEqual(len(frames[2]),10)
            self.assertEqual(len(frames[3]),10)

    def test_values_filtering_denominators_and_scaling(self):
        x=sparse.csr_matrix([[10.,0,2],[0,4,3],[2,2,0],[4,0,8],[6,0,4],[8,0,0]])
        np.testing.assert_array_equal(prep.retained_pair_values(x,[0,1,2]).toarray(),x.toarray())
        means,ids=collapsed_average(x,np.array([0,0,1]),2)
        np.testing.assert_array_equal(ids,[0,1])
        np.testing.assert_array_equal(means.toarray(),np.column_stack((x[:,:2].toarray().mean(axis=1),x[:,2].toarray().ravel())))
        for invalid in (-x,sparse.csr_matrix([[np.nan]]),sparse.csr_matrix([[0.]])):
            with self.assertRaises(ValueError): prep.retained_pair_values(invalid,np.arange(invalid.shape[1]))
        mask,_,_=prep.prevalence_mask(x[:4],.5,1)
        changed=x.toarray(); changed[4:]=100
        np.testing.assert_array_equal(mask,prep.prevalence_mask(sparse.csr_matrix(changed[:4]),.5,1)[0])
        y=np.array([0,1,0,1])
        model=new_model(); model.fit(x[:4],y)
        np.testing.assert_allclose(model[0].scale_,np.sqrt(x[:4].toarray().var(axis=0)))
        np.testing.assert_array_equal(predict_model(x[:4,:0],y,x[4:,:0]),[.5,.5])

    def test_complete_family_corrections_ranking_and_constants(self):
        p=np.array([.001,.01,.03,.2])
        q=mw.multiple_testing(p)
        np.testing.assert_allclose(q['by'],np.minimum(1,q['bh']*sum(1/np.arange(1,5))))
        self.assertTrue(np.all((q['by']<=.1) <= (q['bh']<=.1)))
        x=sparse.csr_matrix([[0,0,1],[0,0,1],[2,2,1],[3,3,1]])
        columns,table=mw.mw_ranked_family(x,np.array([0,0,1,1]),[9,2,7],'pair',[1,1,1])
        np.testing.assert_array_equal(columns,[1,0])
        np.testing.assert_array_equal(table.feature_id,[2,9])
        metrics,pred=evaluate(x[:2,:0],x[2:,:0],np.array([0,1]),np.array([0,1]),[1,10])
        self.assertTrue(metrics.effective_k.eq(0).all())
        self.assertTrue(pred.probability.eq(.5).all())

    def test_pairwise_sign_self_exclusion_constants_and_guard(self):
        x=sparse.csr_matrix([[1,-1,0],[0,0,1],[-1,1,0],[0,0,-1]])
        self.assertAlmostEqual(u.average_pairwise_absolute_correlation(x,[0,1,2])['average_pairwise_absolute_correlation'],1/3)
        self.assertAlmostEqual(u.average_pairwise_absolute_correlation(x,[0,1])['average_pairwise_absolute_correlation'],1)
        for columns in ([],[0]): self.assertEqual(u.average_pairwise_absolute_correlation(x,columns)['status'],'undefined_subset_size')
        with patch.object(sparse.csr_matrix,'toarray',side_effect=AssertionError('Dense allocation')):
            self.assertEqual(u.average_pairwise_absolute_correlation(sparse.csr_matrix((2,10001)),range(10001))['status'],'skipped_size_limit')
        with self.assertRaises(ValueError): u.average_pairwise_absolute_correlation(sparse.csr_matrix([[1,0],[1,2]]),[0,1])
        self.assertEqual(u.average_pairwise_absolute_correlation(x,[0,1,2]),u.average_pairwise_absolute_correlation(x,[2,0,1]))

    def test_shuffled_test_formula_boundaries_and_class_counts(self):
        counts=np.array([12,17,20,15,16]); n,r=1000,17
        result=discovery_statistics(counts,r,n)
        self.assertEqual(result['exceedances'],2)
        self.assertEqual(result['p_value'],3/6)
        self.assertEqual(result['log_p_value'],np.log(.5))
        self.assertEqual(discovery_statistics(counts,r,100000)['p_value'],.5)
        self.assertGreater(discovery_statistics([50,51,49],10,n)['p_value'],.5)
        for values,real,size in [([0,0],0,0),([0,0],0,100),([100,100],100,100)]:
            self.assertEqual(discovery_statistics(values,real,size)['p_value'],1)
        self.assertEqual(discovery_statistics([0]*999,100000,1000000)['p_value'],.001)
        self.assertEqual(discovery_statistics([0]*980+[2]*19,1,100)['p_value'],.020)
        self.assertEqual(discovery_statistics([0,9],0,100)['p_value'],1)
        self.assertEqual(discovery_statistics([0],1,100)['p_value'],.5)
        for bad in ([0,.5],[],[-1,2],[1001,0],[np.nan,1]):
            with self.assertRaises(ValueError): discovery_statistics(bad,10,n)
        y=np.tile([0,1],15)
        first=label_permutations(y,'fixture',100)
        np.testing.assert_array_equal(first,label_permutations(y,'fixture',100))
        np.testing.assert_array_equal(first.sum(axis=1),np.repeat(y.sum(),100))
        np.testing.assert_array_equal(first,label_permutations(y,'fixture',999)[:100])

    def test_permutation_count_test_exhaustive_correlated_null(self):
        # Enumerate all label assignments: duplicated columns produce large tied count jumps.
        x=sparse.csr_matrix(np.repeat([[0],[0],[1],[1],[2],[2]],4,axis=1))
        labels=np.array([np.isin(np.arange(6),s).astype(int) for s in itertools.combinations(range(6),3)])
        counts=permutation_discoveries(x,labels)
        for correction in ('bh','by'):
            values=counts[counts.correction.eq(correction)].discoveries.to_numpy()
            p=np.array([np.mean(values>=r) for r in values])
            for threshold in np.unique(p):self.assertLessEqual(np.mean(p<=threshold),threshold+1e-14)
            for i,r in enumerate(values):
                # Excluding this observed labeling makes +1 exactly reproduce the complete orbit.
                self.assertAlmostEqual(discovery_statistics(np.delete(values,i),r,4)['p_value'],p[i])

    def test_shuffle_replay_ignores_column_order_but_not_count_changes(self):
        real=pd.DataFrame([dict(method='bacteria',correction=c,tested_features=10,real_discoveries=2)
            for c in ('bh','by')])
        computed=pd.DataFrame([dict(permutation=i,correction=c,discoveries=i,method='bacteria')
            for i in range(5) for c in ('bh','by')])
        cached=computed[['permutation','method','correction','discoveries']].iloc[::-1]
        actual=runner.checked_shuffle_counts(computed,real,5)
        expected=runner.checked_shuffle_counts(cached,real,5)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
        cached=cached.copy();cached.loc[0,'discoveries']=7
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(actual,runner.checked_shuffle_counts(cached,real,5),check_dtype=False)

    def test_reduced_shuffle_selectors_use_by_only(self):
        from shuffled_fdp_test import statistic
        x=sparse.csr_matrix([[0,1],[1,2],[0,3],[4,0],[2,1],[3,0]])
        y=np.array([[0,0,0,1,1,1],[1,0,1,0,1,0]])
        original=stats.false_discovery_control
        with patch.object(statistic.stats,'false_discovery_control',wraps=original) as correction:
            result=runner.shuffled_counts('bacteria',dict(features=x),y)
        self.assertEqual(set(result.correction),{'by'})
        self.assertTrue(all(call.kwargs['method']=='by' for call in correction.call_args_list))
        for group in ('shoe','raw_pairs','mw_ebh','hurdle_product','hurdle_average'):
            with self.assertRaises(ValueError):runner.shuffled_counts(group,{},y)

    def test_previous_normal_formula_remains_a_separate_reference(self):
        values=np.array([12,17,20,15,16]); r,n=25,1000
        result=normal_approximation_statistics(values,r,n)
        pooled=(r+values.sum())/(n*(len(values)+1))
        variance=pooled*(1-pooled)*(1/n+1/(n*len(values)))
        z=(r/n-values.mean()/n)/np.sqrt(variance)
        self.assertAlmostEqual(result['log_p_value'],stats.norm.logsf(z))
        self.assertNotEqual(result['p_value'],discovery_statistics(values,r,n)['p_value'])
        self.assertTrue(normal_approximation_statistics([0]*999,100000,1000000)['p_value_underflow'])
        for counts,real,size in [([0,0],0,0),([0,0],0,100),([100,100],100,100)]:
            self.assertEqual(normal_approximation_statistics(counts,real,size)['p_value'],1)

    def test_p_value_publisher_updates_only_current_views_without_archives(self):
        from contextlib import ExitStack
        from scripts import plot_comparison as plotting
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); tables=root/'tables'; output=root/'outputs'; work=root/'shuffled_fdp_test/permutation_v1'
            preview=root/'evalue_power/preview'
            for directory in (tables,output,work,preview):directory.mkdir(parents=True)
            rows=[dict(dataset='toy',method=m,correction=c,samples=6,
                **discovery_statistics([0]*999,1,10))
                for m,c in plotting.STAT_ORDER+[('exact_categorical','native')]]
            original=pd.DataFrame(rows)
            table=pd.DataFrame(rows).set_index(['method','correction']).loc[plotting.SHUFFLED_ORDER].reset_index()
            table.assign(normal_log_p_value=[-10000.,-100.,-2.,-.1]).to_csv(work/'p_value_comparison.csv',index=False)
            main=original[original.method.isin(cache.METHODS)]
            main.to_csv(tables/'statistics_comparison.csv',index=False)
            original_statistics=(tables/'statistics_comparison.csv').read_bytes()
            main.to_csv(root/'shuffled_fdp_test/p_values.csv',index=False)
            original.to_csv(preview/'statistics.csv',index=False)
            (output/'negative_log10_p_overview.png').write_bytes(b'old figure')
            unrelated=output/'real_discoveries_overview.png';unrelated.write_bytes(b'untouched')
            cache.seal(work/'complete.json','fixture',[])
            for name,value in dict(CACHE=root,TABLES=tables,OUTPUT=output,SHUFFLED=work).items():
                stack.enter_context(patch.object(plotting,name,value))
            stack.enter_context(patch.object(plotting,'permutation_p_values',return_value=table))
            def draw(*args,output=None,**kwargs):
                self.assertEqual(kwargs['fields'],['negative_log10_p'])
                self.assertIn('normal_log_p_value',args[0])
                ((plotting.OUTPUT if output is None else output)/'negative_log10_p_overview.png').write_bytes(b'new figure')
            stack.enter_context(patch.object(plotting,'statistical_figures',side_effect=draw))
            plotting.publish_shuffled_p_values()
            snapshot={p.name:p.read_bytes() for p in work.iterdir()}
            plotting.publish_shuffled_p_values()
            self.assertEqual(snapshot,{p.name:p.read_bytes() for p in work.iterdir()})
            self.assertEqual(unrelated.read_bytes(),b'untouched')
            self.assertEqual((tables/'statistics_comparison.csv').read_bytes(),original_statistics)
            self.assertFalse((preview/'negative_log10_p_overview.png').exists())

    def test_p_value_plot_overlays_saved_logs_without_clipping_or_underflow(self):
        from scripts import plot_comparison as plotting
        rows=[dict(dataset='toy',method=m,correction=c,permutations=999,p_value_method='permutation',
            log_p_value=np.log(.001),normal_log_p_value=old)
            for (m,c),old in zip(plotting.SHUFFLED_ORDER,[-1e7,-100.,-1.,0.])]
        frame=pd.DataFrame(rows)
        later=frame.assign(dataset='later',normal_log_p_value=-1e8)
        with tempfile.TemporaryDirectory() as temporary,patch.object(plotting,'DATASETS',['toy','later']),\
                patch.object(plotting.plt,'close'):
            plotting.statistical_figures(pd.concat([frame,later]),pd.DataFrame(),output=Path(temporary),
                order=plotting.SHUFFLED_ORDER,labels=plotting.SHUFFLED_LABELS,
                colors=dict(plotting.COLORS,exact_categorical='#CCBB44'),fields=['negative_log10_p'])
            fig=plotting.plt.gcf();ax=fig.axes[0]
            self.assertEqual(len(ax.patches),8)
            for i in range(4):
                old,new=ax.patches[2*i:2*i+2]
                self.assertAlmostEqual(old.get_height(),-frame.iloc[i].normal_log_p_value/np.log(10))
                self.assertAlmostEqual(new.get_height(),3.)
                self.assertEqual(old.get_alpha(),.25)
                self.assertIsNone(old.get_hatch());self.assertIsNone(new.get_hatch())
                self.assertAlmostEqual(old.get_x()+old.get_width()/2,new.get_x()+new.get_width()/2)
            self.assertEqual(ax.get_yscale(),'symlog')
            self.assertGreater(ax.get_ylim()[1],1e8/np.log(10))
            self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()],
                ['Permutation p-value (solid narrow bar, foreground)',
                 'Binomial/Normal p-value (transparent wide bar, background)'])
            self.assertEqual(fig._suptitle.get_text(),
                '-log10(o-value); 999 shuffles\nPermutation vs Binomial/Normal approximation')
        plotting.plt.close(fig)

    def test_statistic_only_compatibility_is_exact_and_read_only(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'statistic.py'; source.write_text('before')
            before=cache.file_hash(source); source.write_text('after'); after=cache.file_hash(source)
            directory=root/'cv'; directory.mkdir(); protocol=directory/'protocol.json'
            stored=dict(signature='old',stage='cv_baselines',seed=42,permutations=100,code_sha256={'statistic.py':before})
            current=dict(stored,signature='new',permutations=999,code_sha256={'statistic.py':after})
            cache.write_json(protocol,stored)
            record=dict(code_replacements={'statistic.py':dict(before=before,after=after)},
                settings_replacements={'permutations':dict(before=100,after=999)},
                protocols={'cv/protocol.json':dict(sha256=cache.file_hash(protocol),signature='old')})
            cache.write_json(root/'provenance/permutation_p_compatibility.json',record)
            for fold in range(5):cache.seal(directory/f'toy/fold_{fold}/complete.json','old',[])
            with patch.object(cache,'PROJECT',root),patch.object(cache,'PROVENANCE',root/'provenance'),\
                    patch.object(cache,'CV',directory),patch.object(cache,'DATASETS',['toy']),\
                    patch.object(cache,'build_protocol',return_value=current):
                self.assertEqual(cache.open_protocol(directory,'cv_baselines'),'old')
                self.assertEqual(json.loads(protocol.read_text()),stored)
                with patch.object(cache,'build_protocol',return_value=dict(current,seed=43)):
                    with self.assertRaises(ValueError):cache.open_protocol(directory,'cv_baselines')
                with patch.object(cache,'build_protocol',return_value=dict(current,permutations=998)):
                    with self.assertRaises(ValueError):cache.open_protocol(directory,'cv_baselines')
                source.write_text('unapproved')
                with self.assertRaises(ValueError):cache.open_protocol(directory,'cv_baselines')
                source.write_text('after'); (directory/'toy/fold_0/complete.json').unlink()
                with self.assertRaisesRegex(ValueError,'completed historical'):cache.open_protocol(directory,'cv_baselines')

    def test_single_shuffle_budget_and_arbitrary_checkpoint_resumption(self):
        from contextlib import ExitStack
        from scripts import plot_comparison as plotting
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary); base=root/'base'; native=root/'native'; inputs=root/'inputs'
            work=root/'test'; provenance=root/'provenance'
            groups=runner.SHUFFLE_GROUPS; y=np.array([0,0,0,1,1,1])
            labels=label_permutations(y,'toy/full_data',9)
            for path in (base/'toy',native/'toy',inputs/'toy',root/'scripts'):path.mkdir(parents=True)
            (root/'scripts/search_evalue_power.py').write_text('fixture')
            for path in (base/'protocol.json',provenance/'native_run/protocol.json',provenance/'permutation_p_compatibility.json'):
                cache.write_json(path,{})
            pd.DataFrame(dict(sample_id=list('abcdef'),subject_id=list('abcdef'),y=y)).to_csv(inputs/'toy/samples.csv',index=False)
            for parent in (base,native):
                (parent/'toy/statistics.csv').write_text('fixture')
                cache.seal(parent/'toy/complete.json','fixture',[parent/'toy/statistics.csv'])
            for method in runner.METHODS:
                (base/'toy'/f'{method}_features.npz').write_text('fixture')
            for family,names in [('shoe',['preprocessing','membership','evidence_bins']),('raw_pairs',['preprocessing','features'])]:
                path=native/'toy'/family;path.mkdir(exist_ok=True)
                for name in names:(path/f'{name}.npz').write_text('fixture')
            exact=root/'evalue_power/bins/toy';exact.mkdir(parents=True)
            for name in ('complete.json','evidence.npz','comparison.csv'):(exact/name).write_text('fixture')
            def metadata(group):
                methods=[group]
                corrections=['by'] if group in runner.METHODS else ['native']
                return pd.DataFrame([dict(method=m,correction=c,samples=6,tested_features=10,real_discoveries=1)
                    for m in methods for c in corrections])
            def fits(group,p,targets):
                return pd.DataFrame([dict(permutation=i,method=row.method,correction=row.correction,discoveries=int(target[0]))
                    for i,target in enumerate(targets) for row in metadata(group).itertuples()])
            def frozen(dataset,group):return {},metadata(group)
            for name,value in dict(PROJECT=root,CACHE=root,SOURCE=inputs,OUTPUT=base,NATIVE_FULL=native,
                    PROVENANCE=provenance,SHUFFLED=work,DATASETS=['toy'],SHUFFLE_COUNT=9,SHUFFLE_BATCH=2).items():
                stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch.object(runner,'open_protocol',return_value='fixture'))
            frozen_mock=stack.enter_context(patch.object(runner,'shuffle_group_inputs',side_effect=frozen))
            fits_mock=stack.enter_context(patch.object(runner,'shuffled_counts',side_effect=fits))
            publisher=stack.enter_context(patch.object(plotting,'publish_shuffled_p_values'))
            calls=[]
            def interrupted(group,p,targets):
                calls.append(targets)
                if len(calls)==2:raise RuntimeError('fixture interruption')
                return fits(group,p,targets)
            with patch.object(runner,'shuffled_counts',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'):runner.run_shuffled_test()
            self.assertEqual(len(pd.read_csv(work/'toy'/groups[0]/'counts.csv')),2)
            runner.run_shuffled_test()
            self.assertEqual(len(pd.read_csv(work/'p_values.csv')),4)
            comparison=pd.read_csv(work/'p_value_comparison.csv')
            self.assertEqual(len(comparison),4)
            self.assertTrue(comparison.permutations.eq(9).all())
            self.assertIn('normal_log_p_value',comparison)
            for group in groups:
                counts=pd.read_csv(work/'toy'/group/'counts.csv');self.assertEqual(counts.permutation.max(),8)
                pd.testing.assert_frame_equal(runner.checked_shuffle_counts(counts,metadata(group),9),
                    runner.checked_shuffle_counts(fits(group,{},labels),metadata(group),9))
                self.assertFalse((work/'toy'/group/'pilot.json').exists())
            np.testing.assert_array_equal(fits_mock.call_args_list[0].args[2],labels[2:4])
            fits_mock.reset_mock();frozen_mock.reset_mock();runner.run_shuffled_test()
            fits_mock.assert_not_called();frozen_mock.assert_not_called();self.assertEqual(publisher.call_count,2)
            with patch.object(runner,'SHUFFLE_COUNT',11):
                with self.assertRaises(ValueError):runner.run_shuffled_test()
            (inputs/'toy/samples.csv').write_text('changed')
            with self.assertRaises(ValueError):runner.run_shuffled_test()

    def test_folds_and_validation_isolation(self):
        y=np.tile([0,1],20)
        coverage=np.zeros(len(y),int)
        x=sparse.csr_matrix(np.column_stack((np.arange(40)+1,np.r_[np.zeros(32),np.ones(8)])))
        for a,b in study.make_folds(y):
            self.assertEqual(len(np.intersect1d(a,b)),0)
            coverage[b]+=1
        np.testing.assert_array_equal(coverage,np.ones(len(y)))
        ids,_=study.training_pairs(x,np.arange(32))
        np.testing.assert_array_equal(ids,[0])

    def test_cache_rejection_and_resume_without_computation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); case=root/'toy/fold_0'; case.mkdir(parents=True)
            payload=case/'selection.npz'; np.savez_compressed(payload,native=np.array([],int))
            marker=case/'complete.json'; cache.seal(marker,'current',[payload])
            with patch.object(study,'NATIVE_CV',root),patch.object(study,'prepared',side_effect=AssertionError('Recomputed')):
                study.prediction_phase('toy',0,None,None,None,None,None,None,'current')
            with self.assertRaises(ValueError): cache.complete(marker,'old')
            payload.write_bytes(b'changed')
            with self.assertRaises(ValueError): cache.complete(marker,'current')


if __name__=='__main__': unittest.main()

class FullOrbitTests(unittest.TestCase):
    def test_all_small_columns_and_label_orbits(self):
        for m in range(1, 7):
            states = np.array(list(itertools.product([0, 1, 2], repeat=m))).T
            labels = np.array(list(itertools.product([0, 1], repeat=m)))
            scores = .7*(states == 1)-1.1*(states == 2)
            contrasts = np.broadcast_to([.7, -1.1], (len(labels), states.shape[1], 2))
            result = shoe.log_evidence(sparse.csc_matrix(states), contrasts, labels)
            observed = labels@scores
            present = np.any(states != 0, axis=0)
            for k in range(m+1):
                group = labels.sum(axis=1) == k
                expected = observed[group]-logsumexp(observed[group], axis=0)+np.log(group.sum())
                np.testing.assert_allclose(result[group][:, present], expected[:, present], atol=2e-13)
                np.testing.assert_allclose(np.exp(result[group][:, present]).mean(axis=0), 1., atol=2e-13)
            self.assertTrue(np.isneginf(result[:, ~present]).all())


    def test_extremes_duplicates_empty_and_constants(self):
        states = sparse.csc_matrix([[0, 0, 1, 0], [1, 1, 1, 0], [2, 2, 1, 0], [0, 0, 1, 0]])
        labels = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
        scores = np.broadcast_to([1000., -1000.], (2, 4, 2))
        q = shoe.log_evidence(states, scores, labels)
        self.assertTrue(np.isfinite(q[:, :3]).all())
        np.testing.assert_array_equal(q[:, 0], q[:, 1])
        np.testing.assert_allclose(q[:, 2], 0., atol=1e-12)
        self.assertTrue(np.isneginf(q[:, 3]).all())
        self.assertEqual(shoe.log_evidence(sparse.csc_matrix((4, 0)), np.empty((2, 0, 2)), labels).shape, (2, 0))
        order, selected, _ = shoe.native_selection(q[0], np.arange(4))
        np.testing.assert_array_equal(shoe.native_selection(q[0], np.arange(4))[0], order)
        np.testing.assert_array_equal(shoe.native_selection(q[0], np.arange(4))[1], selected)


    def test_batch_independence_and_score_origin(self):
        rng = np.random.default_rng(42)
        states = sparse.csc_matrix(rng.integers(0, 3, size=(15, 30)))
        y = rng.integers(0, 2, size=(3, 15))
        r = rng.normal(size=(3, 30, 2))
        together = shoe.log_evidence(states, r, y)
        for i in range(3):
            np.testing.assert_allclose(together[i], shoe.log_evidence(states, r[i:i+1], y[i:i+1])[0])
        with self.assertRaises(ValueError):
            shoe.log_evidence(states, r, np.full((3, 15), 2))
        with self.assertRaises(ValueError):
            shoe.log_evidence(states*3, r, y)
        with self.assertRaises(ValueError):
            shoe.log_evidence(states, r*np.nan, y)


class StandaloneTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(42)
        values = rng.binomial(1, .35, size=(60, 8))*rng.uniform(.2, 2, size=(60, 8))
        values[:, 0] = 1
        return sparse.csr_matrix(values), np.tile([0, 1], 30)


    def test_fresh_fit_and_training_only_representation(self):
        raw, y = self.fixture()
        with self.assertRaises(ValueError): shoe.fit(sparse.csr_matrix(raw.shape), y, 'invalid')
        a, b = shoe.honest_split(np.arange(len(y)), 'fresh')
        original = shoe.prepare(raw, a, b)
        with (patch('methods.selectors.mann_whitney.mann_whitney_many', side_effect=AssertionError('MW called')),
              patch('methods.selectors.mann_whitney.multiple_testing', side_effect=AssertionError('BH/BY called'))):
            result = shoe.fit(raw, y, 'fresh')
        np.testing.assert_array_equal(result['feature_ids'], original['feature_ids'])
        np.testing.assert_array_equal(result['features'].toarray(), original['features'].toarray())
        np.testing.assert_array_equal(result['features'].toarray(), raw[:, result['feature_ids']].toarray())
        np.testing.assert_array_equal(shoe.transform(raw, result).toarray(), result['features'].toarray())
        scores = shoe.histogram_scores(original['membership'], original['bin_counts'], y[None, a])
        expected = shoe.log_evidence(original['evidence_bins'], scores, y[None, b])[0]
        np.testing.assert_allclose(result['log_evidence'], expected)
        np.testing.assert_array_equal(result['selected_pair_ids'], result['feature_ids'][result['native']])
        repeat = shoe.fit(raw, y, 'fresh')
        np.testing.assert_array_equal(repeat['order'], result['order'])
        np.testing.assert_array_equal(repeat['log_evidence'], result['log_evidence'])


    def test_evidence_labels_cannot_change_learned_scores(self):
        raw, y = self.fixture()
        before = [v.copy() for v in (raw.data, raw.indices, raw.indptr)]
        result = shoe.fit(raw, y, 'fresh')
        changed = y.copy(); changed[result['evidence_indices']] = 1-changed[result['evidence_indices']]
        other = shoe.fit(raw, changed, 'fresh')
        np.testing.assert_array_equal(result['contrasts'], other['contrasts'])
        np.testing.assert_array_equal(result['positive_median'], other['positive_median'])
        np.testing.assert_array_equal(result['feature_ids'], other['feature_ids'])
        for expected, actual in zip(before, (raw.data, raw.indices, raw.indptr)):
            np.testing.assert_array_equal(expected, actual)
        for factor in [.01, 100.]:
            scaled = shoe.fit(raw*factor, y, 'fresh')
            np.testing.assert_array_equal(scaled['feature_ids'], result['feature_ids'])
            np.testing.assert_array_equal(scaled['features'].toarray(), raw[:, result['feature_ids']].toarray()*factor)


    def test_empty_constants_invalid_inputs_and_finite_unclipped_scores(self):
        result = shoe.fit(sparse.csr_matrix(np.ones((20, 2))), np.tile([0, 1], 10), 'constant')
        self.assertEqual(len(result['native']), 0)
        self.assertEqual(result['features'].shape, (20, 0))
        raw, y = self.fixture()
        with self.assertRaises(ValueError): shoe.fit(-raw, y, 'invalid')
        with self.assertRaises(ValueError): shoe.fit(raw, np.ones(len(y))*2, 'invalid')
        trained = shoe.fit(raw, y, 'fresh')
        with self.assertRaises(ValueError): shoe.transform(sparse.csr_matrix((2, raw.shape[1])), trained)
        with self.assertRaises(ValueError): shoe.transform(raw[:, :-1], trained)
        membership = sparse.csr_matrix((np.ones(100, np.int8), (np.arange(100), np.zeros(100, int))), shape=(200, 2))
        labels = np.r_[np.ones(100), np.zeros(100)][None]
        scores = shoe.histogram_scores(membership, np.array([100, 0]), labels)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0, 0, 0], 5)


    def test_uniform_selection_ties_and_empty_family(self):
        order, native, weighted = shoe.native_selection(np.log([30., 30., 1.]), np.array([9, 2, 4]))
        np.testing.assert_array_equal(order, [1, 0, 2])
        np.testing.assert_array_equal(native, [1, 0])
        np.testing.assert_allclose(weighted, np.log([10., 10., 1/3]))
        for values in shoe.native_selection(np.array([]), np.array([], int)):
            self.assertEqual(len(values), 0)
        with self.assertRaises(ValueError): shoe.native_selection(np.array([np.nan]), [0])


class NewSelectorTests(unittest.TestCase):
    def test_hurdle_against_exact_label_and_abundance_orbits(self):
        from scipy.stats import hypergeom
        values = np.array([[0., 0., 1., 2.], [0., 1., 2., 2.], [1., 2., 3., 2.],
                           [2., 4., 4., 0.], [4., 0., 5., 0.], [8., 0., 6., 0.]])
        labels = np.array([np.isin(np.arange(6), chosen).astype(int)
                           for chosen in itertools.combinations(range(6), 3)])
        result = selectors.hurdle_components(sparse.csc_matrix(values), labels)
        for j, column in enumerate(values.T):
            present = column > 0
            k = labels[:, present].sum(axis=1)
            variance = k.var()
            self.assertAlmostEqual(variance, hypergeom.var(6, 3, present.sum()))
            if variance:
                lam = 1/np.sqrt(variance)
                expected = np.logaddexp(lam*k-logsumexp(lam*k)+np.log(len(k)),
                                       -lam*k-logsumexp(-lam*k)+np.log(len(k)))-np.log(2)
                np.testing.assert_allclose(result['log_presence'][:, j], expected, atol=1e-12)
            abundance = column[present]
            for b, y in enumerate(labels):
                k1, s = int(k[b]), len(abundance)
                if not k1 or k1 == s or np.var(abundance) == 0:
                    self.assertEqual(result['log_abundance'][b, j], 0.)
                    continue
                comparisons = []
                for chosen in itertools.combinations(range(s), k1):
                    mask = np.isin(np.arange(s), chosen)
                    comparisons.append(abundance[mask].mean()-abundance[~mask].mean())
                comparisons = np.asarray(comparisons)
                sigma = comparisons.std()
                self.assertAlmostEqual(sigma**2, abundance.var(ddof=1)*(1/k1+1/(s-k1)))
                self.assertAlmostEqual(result['eta'][b, j], 1/sigma)
                observed = abundance[y[present] == 1].mean()-abundance[y[present] == 0].mean()
                expected = np.logaddexp(observed/sigma-logsumexp(comparisons/sigma),
                                       -observed/sigma-logsumexp(-comparisons/sigma))+np.log(len(comparisons)/2)
                self.assertAlmostEqual(result['log_abundance'][b, j], expected, places=11)
        for evidence in [result['log_presence'], result['log_abundance'], *selectors.combine(result).values()]:
            np.testing.assert_allclose(np.exp(evidence).mean(axis=0), 1., atol=2e-12)
        for factor in [1e-9, 1e9]:
            other = selectors.hurdle_components(sparse.csc_matrix(values*factor), labels)
            np.testing.assert_allclose(other['log_abundance'], result['log_abundance'], atol=1e-12)


    def test_coefficients_mw_ties_and_no_bh_calls(self):
        weights = np.array([[-1000., 1000., 3., -2.], [1., 2., 3., 4.]])
        for degree in range(5):
            expected = logsumexp([weights[:, list(c)].sum(axis=1)
                                 for c in itertools.combinations(range(4), degree)], axis=0)
            np.testing.assert_allclose(selectors.log_coefficients(weights, degree), expected)
        x = sparse.csc_matrix([[0., 0.], [0., 0.], [2., 2.], [3., 3.]])
        y = np.array([[0, 0, 1, 1]])
        from scipy.stats import mannwhitneyu
        with patch('methods.selectors.mann_whitney.multiple_testing', side_effect=AssertionError('BH/BY called')):
            log_e, p, _ = selectors.mw_log_evidence(x, y)
            parts = selectors.hurdle_components(x, y)
        expected = mannwhitneyu([2., 3.], [0., 0.], method='asymptotic').pvalue
        np.testing.assert_allclose(p, expected)
        np.testing.assert_allclose(log_e, np.log(.5)-.5*np.log(p))
        for value in selectors.combine(parts).values():
            np.testing.assert_array_equal(value[:, 0], value[:, 1])
            order = shoe.native_selection(value[0], np.array([9, 2]))[0]
            np.testing.assert_array_equal(order, [1, 0])


    def test_full_training_filtering_constants_cache_and_signals(self):
        y = np.repeat([0, 1], 30)
        x = sparse.csr_matrix(np.column_stack((1+5*y, y, np.ones(60), np.arange(60)+1)))
        p = selectors.prepare(x, np.arange(40))
        changed = x.toarray(); changed[40:] *= 20
        other = selectors.prepare(sparse.csr_matrix(changed), np.arange(40))
        np.testing.assert_array_equal(p['feature_ids'], other['feature_ids'])
        np.testing.assert_array_equal(p['features'][:40].toarray(), other['features'][:40].toarray())
        np.testing.assert_array_equal(p['features'].toarray(), x[:, p['feature_ids']].toarray())
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original, changed = parent/'original', parent/'changed'
            original.mkdir(); changed.mkdir()
            labels = y.copy(); labels[40:] = 1-labels[40:]
            before = runner.raw_selections(original, p, y[None], np.arange(4), np.arange(4), 'fixture')
            after = runner.raw_selections(changed, p, labels[None], np.arange(4), np.arange(4), 'fixture')
            for method in before:
                for field in before[method]: np.testing.assert_array_equal(before[method][field], after[method][field])
        with tempfile.TemporaryDirectory() as directory:
            first = selectors.hurdle_components(p['features'], y[None], directory)
            with patch.object(selectors, 'log_coefficients', side_effect=AssertionError('Checkpoint recomputed')):
                second = selectors.hurdle_components(p['features'], y[None], directory)
            for key in first: np.testing.assert_array_equal(first[key], second[key])
            with self.assertRaisesRegex(ValueError, 'Changed Hurdle checkpoint'):
                selectors.hurdle_components(p['features'], 1-y[None], directory)
        for log_e in selectors.combine(first).values():
            self.assertGreater(len(shoe.native_selection(log_e[0], p['feature_ids'])[1]), 0)
        empty = selectors.hurdle_components(sparse.csc_matrix((6, 0)), np.array([[0, 1]*3]))
        self.assertEqual(empty['log_presence'].shape, (1, 0))
        boundary = selectors.hurdle_components(sparse.csc_matrix(np.ones((6, 2))), np.ones((1, 6)))
        np.testing.assert_array_equal(boundary['log_abundance'], 0)
        np.testing.assert_array_equal(boundary['log_presence'], 0)
        with self.assertRaises(ValueError): selectors.hurdle_components(-x, y[None])


class BernoulliJoinMethodTests(unittest.TestCase):
    def test_fixed_threshold_fails_instead_of_raising_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sparse.save_npz(root / "X.npz", sparse.csr_matrix(np.tile([[1],[1],[0],[0]], (1,3))))
            np.savez(root / "split.npz", train_index=np.arange(4), feature_ids=np.arange(3))
            result = subprocess.run([str(JOIN_PYTHON), str(PROJECT / "methods/graphs/worker.py"),
                str(root / "X.npz"), str(root / "split.npz"), str(root / "join"),
                "--threshold", ".7", "--workers", "1", "--memory-gb", "1", "--memory-reserve-gb", "0",
                "--working-memory-gb", "1", "--community-working-memory-gb", "1",
                "--output-gb", "1", "--free-disk-gb", "0", "--community-max-edges", "1",
                "--fixed-threshold"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Fixed threshold 0.70 requires 3 edges", result.stderr)
            self.assertFalse((root / "join/integration.json").exists())


    def test_memory_fallback_chooses_smallest_fitting_threshold(self):
        self.assertEqual(MAX_COMMUNITY_EDGES, 500_000_000)
        self.assertEqual(
            fallback_thresholds(0.7),
            [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        )
        self.assertEqual(
            choose_effective_threshold({0.7: 700, 0.75: 310, 0.8: 250}, 300),
            0.8,
        )
        with self.assertRaisesRegex(RuntimeError, "threshold 1.0"):
            choose_effective_threshold({0.7: 700, 1.0: 301}, 300)


    def test_float32_threshold_boundary_is_counted_consistently(self):
        edges = np.zeros(2, dtype=[
            ("feature_j", "<u4"), ("feature_k", "<u4"), ("phi", "<f4")
        ])
        edges["phi"] = [np.float32(0.9), np.nextafter(np.float32(0.9), 0)]
        manifest = {"edge_weight": {"field": "phi", "transform": "abs"}}
        counts = count_threshold_edges(None, [0.9], manifest, lambda unused: [edges])
        self.assertEqual(counts, {0.9: 1})


    def test_external_exact_worker_matches_brute_force_at_threshold(self):
        detections = np.column_stack([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            np.ones(10, dtype=int),
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path, split_path, output = root / "X.npz", root / "split.npz", root / "result"
            sparse.save_npz(matrix_path, sparse.csr_matrix(detections))
            np.savez_compressed(
                split_path, train_index=np.arange(10), feature_ids=np.arange(5)
            )
            subprocess.run([
                str(JOIN_PYTHON),
                str(PROJECT / "methods/graphs/worker.py"),
                str(matrix_path), str(split_path), str(output),
                "--threshold", "0.6", "--workers", "2",
                "--memory-gb", "1", "--working-memory-gb", "1",
                "--community-working-memory-gb", "1",
                "--output-gb", "1", "--free-disk-gb", "0",
                "--memory-reserve-gb", "0", "--keep-graph-gb", "1",
            ], check=True, capture_output=True, text=True)
            observed = set()
            for shard_path in sorted((output / "graph/edges").glob("part-*.npy")):
                shard = np.load(shard_path)
                observed.update(
                    (min(int(row["feature_j"]), int(row["feature_k"])),
                     max(int(row["feature_j"]), int(row["feature_k"])))
                    for row in shard
                )
            with np.load(output / "assignments.npz") as saved:
                labels = saved["clusters"]
            base_manifest_mtime = (output / "graph/run.json").stat().st_mtime_ns
            resumed = subprocess.run([
                str(JOIN_PYTHON),
                str(PROJECT / "methods/graphs/worker.py"),
                str(matrix_path), str(split_path), str(output),
                "--threshold", "0.7", "--workers", "2",
                "--fixed-threshold", "--community-max-edges", "800000000",
                "--memory-gb", "1", "--working-memory-gb", "1",
                "--community-working-memory-gb", "1",
                "--output-gb", "1", "--free-disk-gb", "0",
                "--memory-reserve-gb", "0", "--keep-graph-gb", "1",
            ], check=True, capture_output=True, text=True)
            self.assertIn("Reusing completed exact graph", resumed.stdout)
            self.assertEqual(
                base_manifest_mtime, (output / "graph/run.json").stat().st_mtime_ns
            )
            with (output / "integration.json").open(encoding="utf-8") as handle:
                integration = json.load(handle)
            self.assertTrue(integration["fixed_threshold"])
            self.assertEqual(integration["community_hard_edge_cap"], 800_000_000)

        expected = set()
        for left in range(4):
            for right in range(left + 1, 4):
                correlation = np.corrcoef(detections[:, left], detections[:, right])[0, 1]
                if abs(correlation) >= 0.6 - 1e-12:
                    expected.add((left, right))
        self.assertEqual(observed, expected)
        self.assertAlmostEqual(
            abs(np.corrcoef(detections[:, 0], detections[:, 1])[0, 1]), 0.6
        )
        self.assertNotIn(labels[4], labels[:4])
        self.assertEqual(integration["requested_threshold"], 0.7)
        self.assertEqual(integration["effective_threshold"], 0.7)


    def test_sparse_packing_matches_detection_matrix(self):
        matrix = sparse.csr_matrix([
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [4.0, 5.0, 0.0],
            [0.0, 6.0, 7.0],
        ])
        with tempfile.TemporaryDirectory() as directory:
            packed = Path(directory) / "packed"
            supports = pack_sparse_detection(matrix, np.array([9, 2, 17]), packed)
            words = np.load(packed / "bits.npy")
            detected = np.unpackbits(
                words.view(np.uint8), axis=1, bitorder="little"
            )[:, : matrix.shape[0]]
        np.testing.assert_array_equal(detected, matrix.toarray().T > 0)
        np.testing.assert_array_equal(supports, [2, 3, 2])


    def test_sparse_packing_rejects_all_zero_feature(self):
        matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "all-zero"):
                pack_sparse_detection(matrix, np.array([0, 1]), Path(directory) / "packed")


    def test_all_one_feature_becomes_singleton(self):
        labels = stable_cluster_labels(
            feature_ids=np.array([20, 10, 30]),
            labels=np.array([7, 7, -1]),
            supports=np.array([2, 2, 4]),
            n=4,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[2], labels[0])
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            stable_cluster_labels([0], [-1], [3], n=4)


    def test_supplied_values_are_mean_aggregated_without_transformation(self):
        train = sparse.csr_matrix([[0.2, 0.4, 0.4], [0.0, 0.6, 0.4]])
        test = sparse.csr_matrix([[0.1, 0.3, 0.6]])
        observed_train, observed_test, sizes = aggregate_pair_clusters(
            train, test, np.array([0, 0, 1])
        )
        np.testing.assert_allclose(
            observed_train.toarray(), [[0.3, 0.4], [0.3, 0.4]]
        )
        np.testing.assert_allclose(observed_test.toarray(), [[0.2, 0.6]])
        np.testing.assert_array_equal(sizes, [2, 1])
