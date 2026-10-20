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
from shuffled_fdp_test.statistic import discovery_statistics, label_permutations


class PipelineTests(unittest.TestCase):
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
            for method in cache.NATIVE:
                case=native/'shoe' if method=='shoe' else native/'raw_pairs'/method
                case.mkdir(exist_ok=True)
                np.savez_compressed(case/'selection.npz',order=np.arange(79,-1,-1),native=np.array([],int))
            for name,value in dict(FULL=root/'full',NATIVE_FULL=root/'native',DATASETS=['toy'],TABLES=root/'tables').items():
                stack.enter_context(patch.object(plotting,name,value))
            actual=plotting.ranked_pairwise_correlations()
            self.assertEqual(len(actual),7*4)
            self.assertTrue(actual.effective_k.eq(actual.requested_k).all())
            for row in actual.itertuples():
                order=np.arange(80) if row.method not in cache.NATIVE else np.arange(79,-1,-1)
                expected=u.average_pairwise_absolute_correlation(x,order[:row.requested_k])
                self.assertAlmostEqual(row.average_pairwise_absolute_correlation,expected['average_pairwise_absolute_correlation'])
            with patch.object(sparse,'load_npz',side_effect=AssertionError('Derived cache recomputed')):
                pd.testing.assert_frame_equal(actual,plotting.ranked_pairwise_correlations())
            np.savez_compressed(native/'raw_pairs/mw_ebh/selection.npz',order=np.arange(80),native=np.array([],int))
            changed=plotting.ranked_pairwise_correlations()
            rows=changed[changed.method.eq('mw_ebh')].set_index('requested_k')
            self.assertAlmostEqual(rows.loc[10,'average_pairwise_absolute_correlation'],u.average_pairwise_absolute_correlation(x,range(10))['average_pairwise_absolute_correlation'])

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
                                  (full,dict(SOURCE=source,OUTPUT=fullroot,NATIVE_FULL=nf,PROVENANCE=root/'provenance',DATASETS=['toy']))]:
                for name,value in values.items(): stack.enter_context(patch.object(module,name,value))
                stack.enter_context(patch.object(module,'open_protocol',side_effect=lambda directory,stage:stage))
                stack.enter_context(patch.object(module,'cluster',return_value=(np.array([0,0,1,2]),dict(
                    clusters=3,effective_threshold=.7,join=dict(edge_count=1),community=dict(deduplicated_edges=1),threshold_edge_counts={'.7':1}))))
            study.main(); full.main()
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
        counts=np.array([12,17,20,15,16]); n,r,m=1000,50,len(counts)
        pooled=(r+counts.sum())/(n*(m+1))
        z=(r/n-counts.sum()/(n*m))/np.sqrt(pooled*(1-pooled)*(1/n+1/(n*m)))
        result=discovery_statistics(counts,r,n)
        self.assertAlmostEqual(result['p_value'],stats.norm.sf(z))
        self.assertGreater(discovery_statistics([50,51,49],10,n)['p_value'],.5)
        for values,real,size in [([0,0],0,0),([0,0],0,100),([100,100],100,100)]:
            self.assertEqual(discovery_statistics(values,real,size)['p_value'],1)
        extreme=discovery_statistics([0]*100,100000,1000000)
        self.assertTrue(extreme['p_value_underflow']); self.assertTrue(np.isfinite(extreme['log_p_value']))
        for bad in ([0,.5],[1],[-1,2],[1001,0]):
            with self.assertRaises(ValueError): discovery_statistics(bad,10,n)
        y=np.tile([0,1],15)
        first=label_permutations(y,'fixture',100)
        np.testing.assert_array_equal(first,label_permutations(y,'fixture',100))
        np.testing.assert_array_equal(first.sum(axis=1),np.repeat(y.sum(),100))

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
