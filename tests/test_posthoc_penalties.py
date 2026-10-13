"""Checks for cache-only uncertainty/reliability penalties."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu
from scripts.run_posthoc_penalty_search import penalty_statistics, select_penalized, logistic_conditional_gains, source_weights_from_assignments, composition_penalties, resampled_l1_frequency, taxon_coverage_from_assignments, mw_effect, GRID
from scripts.run_qinj_redundancy_analysis import new_model


class PosthocPenaltyTests(unittest.TestCase):
    def test_six_method_ablation_retunes_each_component_and_only_removes_requested_gate(self):
        from scripts.compare_current_penalty_ablation import training_choices, METHODS, POWERS
        from scripts.run_joint_k_penalty_selection import K_VALUES
        def profile(candidate):
            rows = []
            for family, powers in {'baseline': [0], 'bacteria': [0], candidate: POWERS}.items():
                for power in powers:
                    for k in K_VALUES:
                        for fold in range(3):
                            auc = .6
                            if family == 'baseline' and k == 2: auc = .7
                            if family == 'bacteria': auc = .99
                            if family == candidate and power == .1 and k == 3: auc = [.69, .8, .8][fold]
                            if family == candidate and power == 0 and k == 4: auc = .72
                            rows.append(dict(family=family, strength=power, requested_k=k,
                                             fold=fold, auroc=auc, effective_k=k))
            return pd.DataFrame(rows)
        power, partition = profile('coverage_power'), profile('coverage_partition_power')
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('No historical outcome input')):
            choices = training_choices(power, partition)
        self.assertEqual(set(choices), set(METHODS))
        self.assertEqual(choices['current_penalty']['family'], 'baseline')
        self.assertEqual(choices['without_part2']['family'], 'baseline')
        self.assertEqual((choices['without_part3']['family'], choices['without_part3']['strength'],
                          choices['without_part3']['requested_k']), ('coverage_partition_power', 0, 4))
        self.assertEqual((choices['without_part5']['family'], choices['without_part5']['strength'],
                          choices['without_part5']['requested_k']), ('coverage_partition_power', .1, 3))
        self.assertEqual(training_choices(power.sample(frac=1, random_state=8), partition.sample(frac=1, random_state=3)), choices)
        partition.loc[partition.family.eq('coverage_partition_power'), 'auroc'] = .5
        self.assertEqual(training_choices(power, partition)['without_part5']['family'], 'baseline')
        with self.assertRaises(ValueError): training_choices(power, partition.iloc[1:])

    def test_subset_discount_csv_round_trip_preserves_scores_and_tie_order(self):
        from io import StringIO
        table=pd.DataFrame(dict(effect=[.3,.7,.2]))
        stats=pd.DataFrame(dict(partition_instability=[.19999999999999998,.6,.14052691721854327],source_coverage=[1.,1.,.5]))
        train=sparse.csr_matrix([[0,1,0],[1,0,1],[2,1,2],[3,0,3]],dtype=float)
        restored=pd.read_csv(StringIO(stats.to_csv(index=False)),float_precision='round_trip')
        np.testing.assert_array_equal(restored,stats)
        order,cost=select_penalized(train,None,table,stats,'coverage_partition_power',0)
        replay_order,replay_cost=select_penalized(train,None,table,restored,'coverage_partition_power',0)
        self.assertEqual(replay_order,order)
        np.testing.assert_array_equal(replay_cost,cost)

    def test_part2_ablation_locks_raw_and_guarded_settings_using_only_cv(self):
        from scripts.analyze_partition_discount_ablation import choose_component_setting
        from scripts.run_joint_k_penalty_selection import K_VALUES
        rows=[]
        for family,powers in {'baseline':[0],'coverage_power':[0,.5]}.items():
            for gamma in powers:
                for k in K_VALUES:
                    for fold in range(3):
                        auc=.6
                        if family=='coverage_power' and gamma==.5 and k==3:auc=[.59,.7,.7][fold]
                        rows.append(dict(family=family,strength=gamma,requested_k=k,fold=fold,auroc=auc,effective_k=k))
        cv=pd.DataFrame(rows)
        with patch('scripts.run_posthoc_penalty_search.cached_root',side_effect=AssertionError('historical cache')):
            setting=choose_component_setting(cv,'coverage_power',[0,.5])
        self.assertEqual((setting['gamma'],setting['cv_k']),(.5,3))
        self.assertEqual(setting['guard_choice']['family'],'baseline')  # Raw nominee retained for the ablation, despite failed guard.
        self.assertEqual(choose_component_setting(cv,'coverage_power',[0])['gamma'],0)
        self.assertEqual(choose_component_setting(cv.sample(frac=1,random_state=3),'coverage_power',[0,.5]),setting)

    def test_aggregate_accepts_explicit_methods_and_rejects_missing_rows(self):
        from scripts.plot_aggregate_coverage_comparison import aggregate
        methods=['baseline','candidate'];datasets=['fresh_a','fresh_b'];ks=[1,2]
        rows=[dict(dataset=d,method=m,requested_k=k,auroc=.6+.2*(d=='fresh_b'),
                   precision=.5,recall=.6,f1=.55) for d in datasets for m in methods for k in ks]
        data=pd.DataFrame(rows)
        mean,sd=aggregate(data,datasets,ks,methods=methods)
        self.assertEqual(len(mean),4)
        np.testing.assert_allclose(mean.auroc,.7)
        np.testing.assert_allclose(sd.auroc,np.sqrt(.02))
        self.assertTrue(mean.n_datasets.eq(2).all())
        with self.assertRaises(ValueError):aggregate(data.iloc[1:],datasets,ks,methods=methods)

    def test_fold_consistency_guard_is_reference_free_and_rejects_fold_losses(self):
        from scripts.run_fold_consistent_partition_transfer import choose_fold_consistent_settings
        from scripts.run_joint_k_penalty_selection import K_VALUES
        grid={'coverage_partition_power':[0, .1]}
        def observations(gains):
            rows=[]
            for family,weights in {'baseline':[0],'bacteria':[0],**grid}.items():
                for strength in weights:
                    for k in K_VALUES:
                        for fold in range(3):
                            auc=.7 if family=='baseline' and k==2 else .6
                            if family=='bacteria':auc=.99  # Reporting reference cannot win.
                            if family=='coverage_partition_power' and strength==.1 and k==3:auc=.7+gains[fold]
                            rows.append(dict(family=family,strength=strength,requested_k=k,fold=fold,auroc=auc,effective_k=k))
            return pd.DataFrame(rows)
        cv=observations([0,0,.036])
        with patch('scripts.run_posthoc_penalty_search.cached_root',side_effect=AssertionError('historical cache')):
            choice=choose_fold_consistent_settings(cv,grid)[2]
            fresh=choose_fold_consistent_settings(cv[cv.family.ne('bacteria')],grid,reference_families=())[2]
        self.assertEqual(choice,fresh)
        self.assertEqual((choice['family'],choice['strength'],choice['requested_k']),('coverage_partition_power',.1,3))
        self.assertTrue(choice['guard_accepted'])
        self.assertFalse(choice['would_pass_one_se'])  # Consistency is not confidence.
        self.assertEqual(choose_fold_consistent_settings(cv.sample(frac=1,random_state=42),grid)[2],choice)
        rejected=choose_fold_consistent_settings(observations([-.01,.03,.03]),grid)[2]
        self.assertEqual((rejected['family'],rejected['requested_k']),('baseline',2))
        self.assertFalse(rejected['guard_accepted'])
        self.assertEqual(choose_fold_consistent_settings(observations([0,0,0]),grid)[2]['family'],'baseline')
        with self.assertRaises(ValueError):choose_fold_consistent_settings(cv.iloc[1:],grid)

    def test_partition_coverage_reuses_training_consistency_and_is_standalone(self):
        from scripts.run_posthoc_penalty_search import coverage_partition_scores
        from sklearn.model_selection import StratifiedKFold
        effect, instability, coverage = np.array([.4, -.3, .1, .8]), np.array([.1, .02, .2, 0]), np.array([.5, 1, .9, 1])
        train = sparse.csr_matrix([[0, 1, 0, 1], [1, 0, 1, 1], [2, 1, 2, 1], [3, 0, 3, 1]], dtype=float)
        table = pd.DataFrame(dict(effect=effect))
        stats = pd.DataFrame(dict(partition_instability=instability, source_coverage=coverage))
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            score, loss = coverage_partition_scores(effect, instability, coverage, 1)
            order, selected_loss = select_penalized(train, None, table, stats, 'coverage_partition_power', 1)
        np.testing.assert_allclose(score[:3], [.15, .28, 0])
        np.testing.assert_allclose(score+loss, np.abs(effect))
        self.assertEqual(order, [1, 0, 2])
        np.testing.assert_allclose(selected_loss, loss[order])
        self.assertEqual(select_penalized(train[:, :0], None, table.iloc[:0], stats.iloc[:0], 'coverage_partition_power', 1)[0], [])
        for bad in ([-1]*4, [np.nan]*4, [0]):
            with self.assertRaises(ValueError):coverage_partition_scores(effect, bad, coverage, 1)
        values=np.random.default_rng(70).uniform(size=(18, 3)); target=np.repeat([0, 1], 9)
        signed=mw_effect(values,target)
        fresh_table=pd.DataFrame(dict(effect=signed, p=.01, pair_count=1))
        fresh_stats=penalty_statistics(sparse.csr_matrix(values),target,fresh_table)
        subset_effects=np.array([np.sign(signed)*mw_effect(values[ids],target[ids])
            for _,ids in StratifiedKFold(3,shuffle=True,random_state=42).split(values,target)])
        expected=np.maximum(0,np.minimum(np.abs(signed),subset_effects.min(axis=0)))
        np.testing.assert_allclose(coverage_partition_scores(signed,fresh_stats.partition_instability,np.ones(3),0)[0],expected)

    def test_bh_coverage_power_restores_original_order_without_historical_inputs(self):
        from scripts.run_posthoc_penalty_search import bh_coverage_power_scores
        table = pd.DataFrame(dict(p=[.001, .01, .02, 0], effect=[.3, .4, .2, 1]))
        stats = pd.DataFrame(dict(source_coverage=[.01, 1, 1, 1]))
        train = sparse.csr_matrix([[0, 1, 0, 1], [1, 0, 1, 1], [2, 1, 2, 1], [3, 0, 3, 1]], dtype=float)
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            score, loss = bh_coverage_power_scores(table.p, stats.source_coverage, 1)
            order, selected_loss = select_penalized(train, None, table, stats, 'bh_coverage_power', 1)
            zero_order, zero_loss = select_penalized(train, None, table, stats, 'bh_coverage_power', 0)
        np.testing.assert_allclose(score[:3], [.03, 2, -np.log10(.02)])
        self.assertTrue(np.isfinite(score).all())  # An underflowed p-value is clamped, not infinite.
        np.testing.assert_allclose(score+loss, -np.log10(np.maximum(table.p, np.finfo(float).tiny)))
        self.assertEqual(order, [1, 2, 0])
        np.testing.assert_allclose(selected_loss, loss[order])
        self.assertEqual(zero_order, [0, 1, 2])
        np.testing.assert_allclose(zero_loss, 0)
        self.assertEqual(select_penalized(train*7, None, table, stats, 'bh_coverage_power', 1)[0], order)
        self.assertEqual(select_penalized(train, None, table, stats, 'bh_coverage_power', 1, maximum=1)[0], [1])
        self.assertEqual(select_penalized(train[:, :0], None, table.iloc[:0], stats.iloc[:0], 'bh_coverage_power', 1)[0], [])
        tied = table.iloc[:2].assign(p=.01)
        self.assertEqual(select_penalized(train[:, :2], None, tied, stats.iloc[:2].assign(source_coverage=1), 'bh_coverage_power', 1)[0], [0, 1])
        for p, r, gamma in (([.01], [0], 1), ([.01], [1.1], 1), ([np.nan], [.5], 1), ([.01, .02], [.5], 1), ([.01], [.5], -1)):
            with self.assertRaises(ValueError):bh_coverage_power_scores(p, r, gamma)

    def test_smooth_snr_coverage_is_standalone_and_exports_actual_loss(self):
        from scripts.run_posthoc_penalty_search import coverage_snr_scores
        effect, se, coverage = np.array([.4, -.3, 0, .8]), np.array([.1, .02, 0, .1]), np.array([.5, 1, .9, 1])
        train = sparse.csr_matrix([[0, 1, 0, 1], [1, 0, 1, 1], [2, 1, 2, 1], [3, 0, 3, 1]], dtype=float)
        table = pd.DataFrame(dict(effect=effect))
        stats = pd.DataFrame(dict(effect_lcb=se, source_coverage=coverage))
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            score, loss = coverage_snr_scores(effect, se, coverage, 1)
            order, selected_loss = select_penalized(train, None, table, stats, 'coverage_snr_power', 1)
        np.testing.assert_allclose(score[:3], [.4*.16/.17*.5, .3*.09/.0904, 0])
        np.testing.assert_allclose(score+loss, np.abs(effect))
        self.assertEqual(order, [1, 0, 2])  # Constant column 3 excluded, even with a large cached effect.
        np.testing.assert_allclose(selected_loss, loss[order])
        np.testing.assert_allclose(coverage_snr_scores(-effect, se, coverage, 1)[0], score)
        np.testing.assert_allclose(coverage_snr_scores(effect, np.zeros(4), coverage, 0)[0], np.abs(effect))
        self.assertEqual(select_penalized(train, None, table, stats, 'coverage_snr_power', 1, maximum=1)[0], [1])
        self.assertEqual(select_penalized(train[:, :0], None, table.iloc[:0], stats.iloc[:0], 'coverage_snr_power', 1)[0], [])
        for bad in ([-1]*4, [np.nan]*4, [0]):
            with self.assertRaises(ValueError):coverage_snr_scores(effect, bad, coverage, 1)
        for bad_power in (-1, np.nan):
            with self.assertRaises(ValueError):coverage_snr_scores(effect, se, coverage, bad_power)

    def test_direction_alignment_coverage_is_a_standalone_effect_discount(self):
        from scripts.run_posthoc_penalty_search import coverage_alignment_scores
        effect, bad_direction, coverage = np.array([.4, -.3, .1]), np.array([.4, .02, 0]), np.array([.5, 1, .9])
        values = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [2, 1, 2], [3, 0, 3]], dtype=float)
        table = pd.DataFrame(dict(effect=effect))
        stats = pd.DataFrame(dict(linear_alignment=bad_direction, source_coverage=coverage))
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            score, loss = coverage_alignment_scores(effect, bad_direction, coverage, 1)
            order, selected_loss = select_penalized(values, None, table, stats, 'coverage_alignment_power', 1)
        np.testing.assert_allclose(score, [0, .28, .09])
        np.testing.assert_allclose(score+loss, np.abs(effect))
        self.assertEqual(order, [1, 2, 0])
        np.testing.assert_allclose(selected_loss, loss[order])
        self.assertEqual(select_penalized(values[:, :0], None, table.iloc[:0], stats.iloc[:0], 'coverage_alignment_power', 1)[0], [])
        for bad in ([-1]*3, [np.nan]*3, [1]*3, [0]):
            with self.assertRaises(ValueError):coverage_alignment_scores(effect, bad, coverage, 1)

    def test_uncertainty_coverage_shrinks_training_effects_without_history(self):
        from scripts.run_posthoc_penalty_search import coverage_uncertainty_scores
        effect, se, coverage = np.array([.4, -.3, .1]), np.array([.1, .02, .2]), np.array([.5, 1, .9])
        train = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [2, 1, 2], [3, 0, 3]], dtype=float)
        table = pd.DataFrame(dict(effect=effect))
        stats = pd.DataFrame(dict(effect_lcb=se, source_coverage=coverage))
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            score, loss = coverage_uncertainty_scores(effect, se, coverage, 1)
            order, selected_loss = select_penalized(train, None, table, stats, 'coverage_uncertainty_power', 1)
        np.testing.assert_allclose(score, [.102, .2608, 0])
        np.testing.assert_allclose(score+loss, np.abs(effect))
        self.assertEqual(order, [1, 0, 2])
        np.testing.assert_allclose(selected_loss, loss[order])
        np.testing.assert_allclose(coverage_uncertainty_scores(effect, np.zeros(3), coverage, 0)[0], np.abs(effect))
        self.assertEqual(select_penalized(train[:, :0], None, table.iloc[:0], stats.iloc[:0], 'coverage_uncertainty_power', 1)[0], [])
        for bad in ([-1]*3, [np.nan]*3, [0]):
            with self.assertRaises(ValueError):coverage_uncertainty_scores(effect, bad, coverage, 1)

    def test_coverage_logistic_gain_uses_training_conditional_gains_from_empty_set(self):
        from scipy.special import ndtri
        values = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [2, 1, 2], [3, 0, 3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        gains = logistic_conditional_gains(values, y, [])
        self.assertTrue(np.isfinite(gains).all())
        self.assertTrue((gains >= 0).all())
        table = pd.DataFrame(dict(effect=[.5, .4, .3], p=[.001]*3))
        stats = pd.DataFrame(dict(source_coverage=[.1, .9, 1]))
        seen = []
        def conditional(train, target, selected):
            np.testing.assert_array_equal(target, y)
            seen.append(list(selected))
            return np.array([3., 2., 1.]) if not selected else np.array([3., 0., 4.])
        with patch('scripts.run_posthoc_penalty_search.logistic_conditional_gains', side_effect=conditional), \
             patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            order, losses = select_penalized(values, y, table, stats, 'coverage_logistic_gain', 1)
        self.assertEqual(order, [1, 2, 0])
        self.assertEqual(seen, [[], [1], [1, 2]])
        np.testing.assert_allclose(losses, ndtri(.001/2)**2-np.array([1.8, 4., .3]))
        self.assertEqual(select_penalized(values[:, :0], y, table.iloc[:0], stats.iloc[:0], 'coverage_logistic_gain', 1)[0], [])

    def test_remaining_catalogue_is_count_only_conservative_and_standalone(self):
        from scripts.run_posthoc_penalty_search import catalogue_fractions_from_assignments, select_remaining_catalogue
        taxa = np.array([0, 0, 0, 0, 1, 1])
        ids, labels = np.arange(6), np.array([0, 0, 1, 1, 2, 2])
        table = pd.DataFrame(dict(feature_id=[0, 1, 2], pair_count=[2, 2, 2], effect=[.8, .7, .3]))
        weights, catalogue = catalogue_fractions_from_assignments(taxa, ids, labels, table)
        np.testing.assert_allclose(catalogue.toarray(), [[.5, 0], [.5, 0], [0, 1]])
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            chosen, losses = select_remaining_catalogue(table.effect, weights, catalogue, 1)
        self.assertEqual(chosen, [0, 2, 1])
        np.testing.assert_allclose(losses, [.4, 0, .525])
        completed, completion_losses = select_remaining_catalogue(table.effect, weights, catalogue, 1, mode='completion')
        self.assertEqual(completed, [0, 1, 2])
        np.testing.assert_allclose(completion_losses, [.4, 0, 0])
        table.attrs['catalogue_metadata'] = (weights, catalogue)
        constant = sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [2, 1, 1], [3, 0, 1]], dtype=float)
        self.assertEqual(select_penalized(constant, None, table, None, 'catalogue_remaining_power', 1)[0], [0, 1])
        static = np.asarray(weights.multiply(catalogue).sum(axis=1)).ravel()
        np.testing.assert_allclose(static, taxon_coverage_from_assignments(taxa, ids, labels, table))
        self.assertEqual(select_remaining_catalogue(table.effect, weights, catalogue, 0)[0], [0, 1, 2])
        self.assertEqual(select_remaining_catalogue(table.effect, weights, catalogue, 1, available=[False, True, True])[0], [1, 2])
        self.assertEqual(select_remaining_catalogue([], sparse.csr_matrix((0, 0)), sparse.csr_matrix((0, 0)), 1)[0], [])
        self.assertEqual(select_remaining_catalogue(table.effect, weights, catalogue, 1, maximum=1)[0], [0])
        # Denominators include original members in unscreened clusters.
        w, c = catalogue_fractions_from_assignments(taxa, ids, labels, table.iloc[[0, 2]])
        np.testing.assert_allclose(c.toarray(), [[.5, 0], [0, 1]])
        for bad in (catalogue*3, catalogue*np.nan):
            with self.assertRaises(ValueError):select_remaining_catalogue(table.effect, weights, bad, 1)
        with self.assertRaises(ValueError):select_remaining_catalogue(table.effect, weights, catalogue, -1)

    def test_bh_coverage_rank_blend_has_exact_standalone_endpoints(self):
        values = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [2, 1, 2], [3, 0, 3]], dtype=float)
        table = pd.DataFrame(dict(effect=[.5, .4, .3]))
        stats = pd.DataFrame(dict(source_coverage=[.001, .01, 1]))
        with patch("scripts.run_posthoc_penalty_search.cached_root", side_effect=AssertionError("historical cache")):
            zero = select_penalized(values, None, table, stats, "bh_coverage_rank_blend", 0)[0]
            full, cost = select_penalized(values, None, table, stats, "bh_coverage_rank_blend", 1)
        self.assertEqual(zero, [0, 1, 2])
        self.assertEqual(full, [2, 1, 0])
        np.testing.assert_allclose(cost, [-1, 0, 1])
        self.assertEqual(select_penalized(values, None, table, stats, "bh_coverage_rank_blend", .5)[0], zero)
        # Blend integer ordinal ranks before normalization so exact rational ties
        # retain BH order even when 1/(p-1) is not exactly representable.
        seven = sparse.csr_matrix(np.outer(np.arange(4), np.arange(1, 8)), dtype=float)
        seven_table = pd.DataFrame(dict(effect=[.5]*7))
        seven_stats = pd.DataFrame(dict(source_coverage=np.arange(1, 8)/7))
        self.assertEqual(select_penalized(seven, None, seven_table, seven_stats,
                                         "bh_coverage_rank_blend", .5)[0], list(range(7)))
        self.assertEqual(select_penalized(values[:, :0], None, table.iloc[:0], stats.iloc[:0], "bh_coverage_rank_blend", .5)[0], [])
        with self.assertRaises(ValueError):
            select_penalized(values, None, table, stats, "bh_coverage_rank_blend", 1.1)

    def test_coverage_support_combines_fresh_training_reliabilities(self):
        from scripts.run_posthoc_penalty_search import coverage_support_power_scores, effective_sample_support
        values = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1], [2, 0, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        table = pd.DataFrame(dict(effect=mw_effect(values, y), pair_count=[4]*3))
        train = sparse.csr_matrix(values)
        stats = pd.DataFrame(dict(concentration=1-effective_sample_support(train)))
        stats["source_coverage"] = [.8, .5, 1]
        support = 1-stats.concentration
        np.testing.assert_allclose(support, [.25, .5, 1])
        with patch("scripts.run_posthoc_penalty_search.cached_root", side_effect=AssertionError("historical cache")):
            score, cost = coverage_support_power_scores(table.effect, stats.source_coverage, support, .5)
            order, selected_cost = select_penalized(train, y, table, stats, "coverage_support_power", .5)
        np.testing.assert_allclose(score, table.effect.abs()*np.sqrt(stats.source_coverage*support))
        np.testing.assert_allclose(score+cost, table.effect.abs())
        self.assertNotIn(2, order)
        np.testing.assert_allclose(selected_cost, cost[order])
        np.testing.assert_allclose(effective_sample_support(train*7), support)
        np.testing.assert_allclose(effective_sample_support(sparse.csr_matrix((4, 1))), 0)
        self.assertEqual(len(effective_sample_support(train[:, :0])), 0)
        for bad in (sparse.csr_matrix((0, 1)), -train, train*np.nan):
            with self.assertRaises(ValueError):
                effective_sample_support(bad)
        self.assertEqual(select_penalized(train[:, :0], y, table.iloc[:0], stats.iloc[:0], "coverage_support_power", 1)[0], [])
        np.testing.assert_allclose(coverage_support_power_scores([.5], [1], [0], 1)[0], 0)
        np.testing.assert_allclose(coverage_support_power_scores([.5], [1], [0], 0)[0], .5)
        for support in ([np.nan]*3, [-.1]*3, [1.1]*3, [.5]):
            with self.assertRaises(ValueError):
                coverage_support_power_scores(table.effect, stats.source_coverage, support, .5)

    def test_dense_k_audit_reproduces_standard_points_without_tuning(self):
        from scripts.audit_dense_k_qinj_penalties import check_standard_points
        dense = pd.DataFrame(dict(requested_k=[1, 2, 3], auroc=[.6, .7, .8],
                                   precision=[.5]*3, recall=[.6]*3, f1=[.55]*3))
        previous = dense.iloc[[0, 2]].copy()
        check_standard_points(dense, previous)
        for bad in (dense.iloc[:2], pd.concat([dense, dense.iloc[:1]]), dense.assign(auroc=np.nan)):
            with self.assertRaises(ValueError):
                check_standard_points(bad, previous)
        with self.assertRaises(AssertionError):
            check_standard_points(dense, previous.assign(auroc=.9))

    def test_coverage_power_is_proportional_standalone_and_handles_constants(self):
        from scripts.run_posthoc_penalty_search import coverage_power_scores
        effect, coverage = np.array([-.8, .5, .99]), np.array([.1, 1, .25])
        train = sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [2, 1, 1], [3, 0, 1]], dtype=float)
        table = pd.DataFrame(dict(effect=effect))
        stats = pd.DataFrame(dict(source_coverage=coverage))
        with patch("scripts.run_posthoc_penalty_search.cached_root", side_effect=AssertionError("historical cache")):
            score, cost = coverage_power_scores(effect, coverage, .5)
            order, selected_cost = select_penalized(train, None, table, stats, "coverage_power", .5)
        np.testing.assert_allclose(score, np.abs(effect)*np.sqrt(coverage))
        np.testing.assert_allclose(score+cost, np.abs(effect))
        self.assertEqual(cost[1], 0)
        self.assertEqual(order, [1, 0])
        np.testing.assert_allclose(selected_cost, cost[order])
        self.assertEqual(select_penalized(train*7, None, table, stats, "coverage_power", .5)[0], order)
        self.assertEqual(select_penalized(train, None, table, stats, "coverage_power", 0)[0], [0, 1])
        np.testing.assert_allclose(coverage_power_scores(effect, coverage, 0)[1], 0)
        self.assertEqual(select_penalized(train, None, table, stats, "coverage_power", .5, maximum=1)[0], [1])
        self.assertEqual(select_penalized(train[:, :0], None, table.iloc[:0], stats.iloc[:0], "coverage_power", .5)[0], [])
        for e, r, gamma in ((effect, coverage, -1), (effect, coverage, np.nan),
                            ([np.nan], [.5], 1), ([.5], [0], 1), ([.5], [1.1], 1),
                            ([.5, .2], [.5], 1)):
            with self.assertRaises(ValueError):
                coverage_power_scores(e, r, gamma)

    def test_bh_log_coverage_preserves_zero_order_and_discounts_incomplete_sources(self):
        train = sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [2, 1, 1], [3, 0, 1]], dtype=float)
        table = pd.DataFrame(dict(p=[.01, .02, .001], effect=[.4, .3, 1]))
        stats = pd.DataFrame(dict(source_coverage=[.001, 1, .5]))
        self.assertEqual(select_penalized(train, None, table, stats, "bh_log_coverage", 0)[0], [0, 1])
        order, penalty = select_penalized(train, None, table, stats, "bh_log_coverage", .25)
        self.assertEqual(order, [1, 0])
        np.testing.assert_allclose(penalty, [0, 3])
        self.assertEqual(select_penalized(train*7, None, table, stats, "bh_log_coverage", .25)[0], order)
        self.assertEqual(select_penalized(train, None, table, stats, "bh_log_coverage", .25, maximum=1)[0], [1])
        self.assertEqual(select_penalized(train[:, :0], None, table.iloc[:0], stats.iloc[:0], "bh_log_coverage", .25)[0], [])
        tied = table.iloc[:2].assign(p=.01)
        self.assertEqual(select_penalized(train[:, :2], None, tied, stats.iloc[:2], "bh_log_coverage", 0)[0], [0, 1])
        for strength in (-1, np.nan):
            with self.assertRaises(ValueError):
                select_penalized(train, None, table, stats, "bh_log_coverage", strength)
        with self.assertRaises(ValueError):
            select_penalized(train, None, table, stats.assign(source_coverage=np.nan), "bh_log_coverage", .25)

    def test_log_coverage_is_standalone_on_unseen_training_inputs(self):
        from scripts.run_posthoc_penalty_search import bh_log_coverage_scores
        taxa = np.array([0, 0, 1, 1, 1, 1])
        ids = np.arange(6)
        labels = np.array([0, 1, 2, 2, 2, 2])
        table = pd.DataFrame(dict(feature_id=[0, 1, 2], pair_count=[1, 1, 4],
                                  p=[.01, .02, .03], effect=[.4, .3, .2]))
        values = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [2, 1, 2], [3, 0, 3]], dtype=float)
        # No benchmarking cache or historical result may be needed by the selector.
        with patch("scripts.run_posthoc_penalty_search.cached_root", side_effect=AssertionError("cache access")):
            coverage = taxon_coverage_from_assignments(taxa, ids, labels, table)
            score, costs = bh_log_coverage_scores(table.p, coverage, 2)
            order, selected_costs = select_penalized(values, None, table,
                pd.DataFrame(dict(source_coverage=coverage)), "bh_log_coverage", 2)
            effect_order, effect_costs = select_penalized(values, None, table,
                pd.DataFrame(dict(effect_log_coverage=-np.log10(coverage))), "effect_log_coverage", 1)
            original_coverage_order, _ = select_penalized(values, None, table,
                pd.DataFrame(dict(taxon_coverage=1-np.sqrt(coverage))), "taxon_coverage", 1)
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(effect_order, order)
        self.assertEqual(original_coverage_order, order)
        np.testing.assert_allclose(effect_costs, costs[order])
        np.testing.assert_allclose(selected_costs, costs[order])
        np.testing.assert_allclose(score, -np.log10(table.p)+2*np.log10(coverage))
        for p, r in (([.01], [0]), ([.01], [1.1]), ([np.nan], [.5]), ([.01, .02], [.5])):
            with self.assertRaises(ValueError):
                bh_log_coverage_scores(p, r, 1)
        self.assertEqual(len(bh_log_coverage_scores([], [], 1)[0]), 0)
        self.assertTrue(np.isfinite(bh_log_coverage_scores([0], [1], 1)[0]).all())

    def test_coverage_uses_unscreened_members_in_denominator_and_handles_mixed_sources(self):
        pair_bacteria=np.array([0,0,0,0,1,1,1,1])
        ids=np.arange(8);labels=np.array([0,0,1,1,2,2,2,2])
        table=pd.DataFrame(dict(feature_id=[0,2],pair_count=[2,4],effect=[.4,.3]))
        coverage=taxon_coverage_from_assignments(pair_bacteria,ids,labels,table)
        np.testing.assert_allclose(coverage,[.5,1])
        mixed=pd.DataFrame(dict(feature_id=[0],pair_count=[8]))
        np.testing.assert_allclose(taxon_coverage_from_assignments(pair_bacteria,ids,np.zeros(8,dtype=int),mixed),[1])
        train=sparse.csr_matrix([[0,1],[1,0],[2,1],[3,0]],dtype=float)
        stats=pd.DataFrame(dict(taxon_coverage=1-np.sqrt(coverage)))
        self.assertEqual(select_penalized(train,None,table,stats,'taxon_coverage',0)[0],[0,1])
        self.assertEqual(select_penalized(train,None,table,stats,'taxon_coverage',1)[0],[1,0])
        stats['cluster_size']=1/np.sqrt(table.pair_count.to_numpy(float))
        order,cost=select_penalized(train,None,table,stats,'size_taxon_coverage',.25)
        expected=table.effect.abs().to_numpy()-.25*stats.cluster_size.to_numpy()-.25*stats.taxon_coverage.to_numpy()
        self.assertEqual(order,np.argsort(-expected,kind='stable').tolist())
        np.testing.assert_allclose(cost,stats.taxon_coverage.to_numpy()[order])
        self.assertEqual(len(taxon_coverage_from_assignments(pair_bacteria,ids,labels,table.iloc[:0])),0)
        with self.assertRaises(ValueError):select_penalized(train,None,table,stats,'taxon_coverage',-.1)

    def test_resampled_selection_frequency_is_deterministic_and_excludes_constants(self):
        rng=np.random.default_rng(42)
        values=np.exp(rng.normal(size=(40,4)));values[:,3]=1
        y=(values[:,0]>np.median(values[:,0])).astype(int)
        train=sparse.csr_matrix(values)
        frequency=resampled_l1_frequency(train,y,1,repeats=6)
        np.testing.assert_array_equal(frequency,resampled_l1_frequency(train,y,1,repeats=6))
        np.testing.assert_array_equal(frequency,resampled_l1_frequency(train*7,y,1,repeats=6))
        self.assertEqual(frequency[3],0)
        self.assertTrue(((frequency>=0)&(frequency<=1)).all())
        self.assertEqual(len(resampled_l1_frequency(train[:,:0],y,1,repeats=6)),0)
        with self.assertRaises(ValueError):resampled_l1_frequency(train,y,0)
        with self.assertRaises(ValueError):resampled_l1_frequency(train,y[:-1],1)
        table=pd.DataFrame(dict(effect=[.3,.4,.25,9],pair_count=[4,4,4,4]))
        stats=pd.DataFrame(dict(cluster_size=[.5]*4,l1_frequency_1=[.9,.1,.8,1]))
        for family in ('l1_stability','size_l1_stability'):
            order,penalty=select_penalized(train,y,table,stats,family,1)
            self.assertEqual(order,[0,2,1]);np.testing.assert_allclose(penalty,[.1,.2,.9])
            self.assertEqual(select_penalized(train[:,:0],y,table.iloc[:0].astype(object),stats.iloc[:0],family,1)[0],[])

    def test_source_complementarity_after_size_uses_only_membership_metadata(self):
        train=sparse.csr_matrix([[0,0,0],[1,2,1],[2,4,0],[3,6,1]],dtype=float)
        table=pd.DataFrame(dict(effect=[.5,.49,.4],pair_count=[4,4,4]))
        table.attrs['source_weights']=sparse.csr_matrix([[1,0],[1,0],[0,1]])
        stats=pd.DataFrame(dict(cluster_size=[.5,.5,.5]))
        self.assertEqual(select_penalized(train,None,table,stats,'size_plus_source',0)[0],[0,1,2])
        order,penalty=select_penalized(train,None,table,stats,'size_plus_source',.25)
        self.assertEqual(order,[0,2,1]);np.testing.assert_allclose(penalty,[0,0,1])
        self.assertEqual(select_penalized(train,None,table,stats,'size_plus_source',.25)[0],order)
        empty=table.iloc[:0].astype(object);empty.attrs['source_weights']=sparse.csr_matrix((0,0))
        self.assertEqual(select_penalized(train[:,:0],None,empty,stats.iloc[:0],'size_plus_source',.25)[0],[])
        with self.assertRaises(ValueError):select_penalized(train,None,table,stats,'size_plus_source',-.1)

    def test_l1_importance_is_training_only_deterministic_and_scale_invariant(self):
        rng=np.random.default_rng(42)
        values=np.exp(rng.normal(size=(50,5)));values[:,4]=1
        y=(values[:,0]+.1*values[:,1]>np.median(values[:,0]+.1*values[:,1])).astype(int)
        train=sparse.csr_matrix(values)
        table=pd.DataFrame(dict(effect=mw_effect(values,y)))
        order,penalty=select_penalized(train,y,table,None,'l1_importance',.1)
        self.assertEqual(len(order),4);self.assertNotIn(4,order)
        self.assertTrue(((penalty>=0)&(penalty<=1)).all())
        self.assertEqual(select_penalized(train,y,table,None,'l1_importance',.1)[0],order)
        scaled,cost=select_penalized(train*7,y,table,None,'l1_importance',.1)
        self.assertEqual(scaled,order);np.testing.assert_allclose(cost,penalty,atol=1e-8)
        score=np.abs(table.effect.to_numpy()[order])-.25*penalty
        self.assertTrue((np.diff(score)<=1e-12).all())
        self.assertEqual(select_penalized(train[:,:0],y,table.iloc[:0].astype(object),None,'l1_importance',.1)[0],[])
        with self.assertRaises(ValueError):select_penalized(train,y,table,None,'l1_importance',0)

    def test_size_plus_corr_uses_selected_only_and_exact_zero_control(self):
        train=sparse.csr_matrix([[0,0,0,1],[1,2,1,1],[2,4,0,1],[3,6,1,1]],dtype=float)
        table=pd.DataFrame(dict(effect=[.5,.49,.4,9],pair_count=[4,4,4,4]))
        stats=pd.DataFrame(dict(cluster_size=[.5,.5,.5,.5]))
        self.assertEqual(select_penalized(train,None,table,stats,'size_plus_corr',0)[0],[0,1,2])
        order,penalty=select_penalized(train,None,table,stats,'size_plus_corr',.25)
        self.assertEqual(order,[0,2,1])
        np.testing.assert_allclose(penalty,[0,1/np.sqrt(5),1])
        self.assertEqual(select_penalized(train,None,table,stats,'size_plus_corr',.25)[0],order)
        self.assertEqual(select_penalized(train[:,:0],None,table.iloc[:0].astype(object),stats.iloc[:0],'size_plus_corr',.25)[0],[])
        with self.assertRaises(ValueError):select_penalized(train,None,table,stats,'size_plus_corr',-.1)

    def test_joint_k_selection_uses_cv_and_rejects_incomplete_grid(self):
        from scripts.run_joint_k_penalty_selection import choose_joint_settings,K_VALUES
        grid={'bh_size_rank_blend':[0,.5]}
        rows=[]
        for family,weights in {'baseline':[0],'bacteria':[0],**grid}.items():
            for weight in weights:
                for k in K_VALUES:
                    auc=.8 if family=='bh_size_rank_blend' and weight==.5 and k==3 else .6
                    for fold in range(3):
                        rows.append(dict(family=family,strength=weight,requested_k=k,fold=fold,auroc=auc,effective_k=k))
        cv=pd.DataFrame(rows)
        mean,locked,choice=choose_joint_settings(cv,grid)
        self.assertEqual((choice['strength'],choice['requested_k']),(.5,3))
        standalone = cv[cv.family.ne('bacteria')].copy()
        with patch('scripts.run_posthoc_penalty_search.cached_root', side_effect=AssertionError('historical cache')):
            fresh_choice = choose_joint_settings(standalone, grid, reference_families=())[2]
        self.assertEqual(fresh_choice, choice)
        self.assertEqual(locked[locked.family.eq('baseline')].requested_k.iloc[0],1)
        self.assertEqual(choose_joint_settings(cv.sample(frac=1,random_state=42),grid)[2],choice)
        cv.loc[cv.family.eq('bh_size_rank_blend'),'auroc']=.6
        self.assertEqual(choose_joint_settings(cv,grid)[2]['strength'],0)
        with self.assertRaises(ValueError):choose_joint_settings(cv.iloc[1:],grid)
        with self.assertRaises(ValueError):choose_joint_settings(pd.concat([cv,cv.iloc[:1]]),grid)

    def test_rank_disagreement_has_exact_bh_and_size_endpoints(self):
        train=sparse.csr_matrix([[0,0,0],[1,2,0],[2,3,1],[3,4,1]],dtype=float)
        table=pd.DataFrame(dict(effect=[.3,.4,.35],pair_count=[1,1,16]))
        stats=pd.DataFrame(dict(cluster_size=1/np.sqrt(table.pair_count)))
        self.assertEqual(select_penalized(train,None,table,stats,'bh_size_rank_blend',0)[0],[0,1,2])
        order,penalty=select_penalized(train,None,table,stats,'bh_size_rank_blend',1)
        self.assertEqual(order,[2,1,0])
        np.testing.assert_allclose(-np.asarray(order)/2-penalty,[0,-.5,-1])
        order,penalty=select_penalized(train,None,table,stats,'bh_size_rank_cap',1)
        self.assertEqual(order,[1,0,2])
        self.assertTrue((penalty>=0).all())
        for family in ('bh_size_rank_blend','bh_size_rank_cap'):
            self.assertEqual(select_penalized(train,None,table,stats,family,0)[0],[0,1,2])
            self.assertEqual(select_penalized(train[:,:0],None,table.iloc[:0].astype(object),stats.iloc[:0],family,.5)[0],[])
            with self.assertRaises(ValueError):select_penalized(train,None,table,stats,family,1.1)

    def test_composition_penalties_count_repetition_not_predictor_values(self):
        taxa=sparse.csr_matrix([[1,0],[.5,.5],[1,0]])
        genes=sparse.csr_matrix([[.25,.25,.25,.25],[.5,.5,0,0],[1,0,0,0]])
        result=composition_penalties([4,4,1],taxa,genes)
        np.testing.assert_allclose(result.effective_genes,[4,2,1])
        np.testing.assert_allclose(result.taxon_concentration,[1,.5,1])
        np.testing.assert_allclose(result.gene_diverse_size,[.5,1,1])
        np.testing.assert_allclose(result.homogeneous_size,[.5,1,1])
        train=sparse.csr_matrix([[0,0,0],[1,2,0],[2,3,1],[3,4,1]],dtype=float)
        table=pd.DataFrame(dict(effect=[.3,.3,.3],p=[.001,.002,.003]))
        for family in ('gene_diverse_size','homogeneous_size'):
            order,penalty=select_penalized(train,None,table,result,family,.25)
            self.assertEqual(order,[0,1,2])
            np.testing.assert_allclose(penalty,[.5,1,1])
        self.assertEqual(len(composition_penalties([],sparse.csr_matrix((0,0)),sparse.csr_matrix((0,0)))),0)
        with self.assertRaises(ValueError):composition_penalties([1,1,1],taxa,genes)
        with self.assertRaises(AssertionError):composition_penalties([4,4,1],taxa*2,genes)

    def test_reliability_discounts_match_formula_and_equal_size_bh_order(self):
        values=np.array([[0,0,0],[1,0,1],[2,0,2],[3,0,3],[4,0,4],[5,5,5]],float)
        y=np.array([0,1,0,1,0,1])
        table=pd.DataFrame(dict(effect=mw_effect(values,y),pair_count=[1,16,16],p=[.001,.002,.004]))
        train=sparse.csr_matrix(values)
        stats=penalty_statistics(train,y,table)
        for family,strength in (("bh_size_shrinkage",1),("bh_support_shrinkage",.2)):
            self.assertEqual(select_penalized(train,y,table,stats,family,0)[0],[0,1,2])
            order,penalties=select_penalized(train,y,table,stats,family,strength)
            count=(table.pair_count.to_numpy(float) if family=="bh_size_shrinkage" else
                   values.sum(axis=0)**2/np.sum(values**2,axis=0))
            pseudo=strength if family=="bh_size_shrinkage" else strength*len(y)
            expected=-np.log10(table.p.to_numpy())*count/(count+pseudo)
            self.assertEqual(order,np.argsort(-expected,kind="stable").tolist())
            np.testing.assert_allclose(-np.log10(table.p.to_numpy()[order])-strength*penalties,expected[order])
            scaled=penalty_statistics(train*7,y,table)
            self.assertEqual(select_penalized(train*7,y,table,scaled,family,strength)[0],order)
            self.assertEqual(select_penalized(train[:,:0],y,table.iloc[:0].astype(object),stats.iloc[:0],family,strength)[0],[])
        self.assertEqual(select_penalized(train,y,table,stats,"bh_size_shrinkage",1)[0],[1,2,0])
        equal=table.copy();equal['pair_count']=1
        self.assertEqual(select_penalized(train,y,equal,stats,"bh_size_shrinkage",16)[0],[0,1,2])

    def test_source_overlap_uses_metadata_and_preserves_zero_strength_bh(self):
        values=np.array([[0,1,1],[1,2,0],[2,0,1],[3,1,0]],float)
        target=np.array([0,1,0,1])
        table=pd.DataFrame(dict(feature_id=[0,1,2],pair_count=[2,1,2],p=[.001,.002,.004],effect=mw_effect(values,target)))
        weights=source_weights_from_assignments(np.array([0,0,0,1,1]),np.arange(5),np.array([0,0,1,2,2]),table)
        np.testing.assert_array_equal(weights.toarray(),[[1,0],[1,0],[0,1]])
        table.attrs["source_weights"]=weights
        train=sparse.csr_matrix(values)
        self.assertEqual(select_penalized(train,target,table,None,"bh_source_overlap",0)[0],[0,1,2])
        chosen,penalties=select_penalized(train,target,table,None,"bh_source_overlap",1)
        self.assertEqual(chosen,[0,2,1]);np.testing.assert_allclose(penalties,[0,0,1])
        self.assertEqual(select_penalized(train,1-target,table,None,"bh_source_overlap",1)[0],chosen)
        wrong=table.copy();wrong.loc[0,"pair_count"]=3
        with self.assertRaises(ValueError):source_weights_from_assignments(np.array([0,0,0,1,1]),np.arange(5),np.array([0,0,1,2,2]),wrong)

    def test_guard_rejects_uncertain_gains_and_retains_baseline_on_ties(self):
        from scripts.run_guarded_size_penalty import choose_guard,bootstrap_differences
        def rows(differences):
            return pd.DataFrame([dict(family=family,strength=0 if family=="baseline" else .25,
                fold=fold,requested_k=k,auroc=.6+(differences[fold] if family=="cluster_size" else 0))
                for family in ("baseline","cluster_size") for fold in range(3) for k in (1,2)])
        self.assertTrue(choose_guard(rows([.04,.04,.04]),(1,2))["apply_penalty"])
        self.assertFalse(choose_guard(rows([-.1,.1,.02]),(1,2))["apply_penalty"])
        self.assertFalse(choose_guard(rows([0,0,0]),(1,2))["apply_penalty"])
        with patch("scripts.run_guarded_size_penalty.REQUIRE_CONSISTENT_FOLDS",True):
            self.assertFalse(choose_guard(rows([-.001,.04,.04]),(1,2))["apply_penalty"])
            self.assertTrue(choose_guard(rows([0,.02,.03]),(1,2))["apply_penalty"])
        with self.assertRaises(ValueError):choose_guard(rows([0,0,0]).iloc[1:],(1,2))
        y=np.array([0,0,1,1]);prediction={1:np.array([.1,.2,.7,.9])}
        uncertainty=bootstrap_differences(prediction,prediction,y,[1])
        self.assertEqual(uncertainty,{"lower":0.,"upper":0.})

    def test_mw_uncertainty_and_ties_match_pairwise_placements(self):
        values = np.array([[0,1],[0,2],[1,2],[2,2],[2,3],[3,4]], float)
        y = np.array([0,1,0,1,0,1])
        effect = mw_effect(values,y)
        expected = 2*mannwhitneyu(values[y==1],values[y==0],axis=0).statistic/9-1
        np.testing.assert_allclose(effect,expected)
        table = pd.DataFrame(dict(effect=effect,pair_count=[1,9],p=[.01,.02]))
        stats = penalty_statistics(sparse.csr_matrix(values),y,table)
        for j in range(2):
            a,b = values[y==1,j],values[y==0,j]
            comparisons = (a[:,None]>b).astype(float)+.5*(a[:,None]==b)
            se = 2*np.sqrt(comparisons.mean(axis=1).var(ddof=1)/3+comparisons.mean(axis=0).var(ddof=1)/3)
            self.assertAlmostEqual(stats.effect_lcb.iloc[j],se)
        self.assertAlmostEqual(stats.cluster_size.iloc[1],1/3)

    def test_constants_empty_candidates_scale_invariance_and_zero_penalty_control(self):
        rng=np.random.default_rng(42)
        values=np.maximum(rng.normal(size=(30,5)),0)
        values[:,4]=1
        y=np.tile([0,1],15)
        table=pd.DataFrame(dict(effect=mw_effect(values,y),pair_count=np.ones(5),p=np.linspace(.01,.05,5)))
        x=sparse.csr_matrix(values)
        stats=penalty_statistics(x,y,table)
        scaled=penalty_statistics(x*7,y,table)
        np.testing.assert_allclose(stats,scaled,atol=1e-12)
        previous=None
        for family in GRID:
            order,penalty=select_penalized(x,y,table,stats,family,0)
            self.assertNotIn(4,order)
            if previous is not None:self.assertEqual(order,previous)
            previous=order
            self.assertEqual(select_penalized(x,y,table,stats,family,0)[0],order)
            empty=table.iloc[:0].astype(object)  # Empty CSV imports have object dtypes.
            self.assertEqual(select_penalized(x[:,:0],y,empty,penalty_statistics(x[:,:0],y,empty),family,.5)[0],[])
        self.assertEqual(select_penalized(x,y,table,stats,"effect_lcb",0,maximum=2)[0],previous[:2])

    def test_bh_penalty_zero_reproduces_p_order(self):
        values=np.array([[0,0],[1,0],[2,1],[3,1],[4,3],[5,4]],float)
        y=np.array([0,1,0,1,0,1])
        table=pd.DataFrame(dict(effect=mw_effect(values,y),pair_count=[1,9],p=[.01,.03]))
        x=sparse.csr_matrix(values)
        stats=penalty_statistics(x,y,table)
        for family in ("bh_cluster_size","bh_instability"):
            self.assertEqual(select_penalized(x,y,table,stats,family,0)[0],[0,1])
        order,_=select_penalized(x,y,table,stats,"bh_cluster_size",1)
        self.assertEqual(order,[1,0])

    def test_conditional_logistic_gain_is_finite_and_training_only(self):
        rng=np.random.default_rng(42)
        values=np.exp(rng.normal(size=(30,4)))
        values[:,3]=1
        y=np.tile([0,1],15)
        table=pd.DataFrame(dict(effect=mw_effect(values,y),pair_count=np.ones(4),p=[.01,.02,.03,1]))
        x=sparse.csr_matrix(values)
        stats=penalty_statistics(x,y,table)
        gains=logistic_conditional_gains(x,y,[0])
        self.assertTrue(np.isfinite(gains).all())
        self.assertTrue((gains>=0).all())
        self.assertEqual(select_penalized(x,y,table,stats,"logistic_gain",0)[0],[0,1,2])
        order,penalties=select_penalized(x,y,table,stats,"logistic_gain",1)
        self.assertEqual(order[0],0)
        self.assertEqual(set(order),{0,1,2})
        self.assertEqual(select_penalized(x,y,table,stats,"logistic_gain",1)[0],order)
        # Independent finite differences of the weighted ridge logistic loss.
        model=new_model();model.fit(x[:,[0]],y)
        design=np.column_stack([np.ones(len(y)),values[:,:2]/values[:,:2].std(axis=0)])
        theta=np.r_[model[-1].intercept_,model[-1].coef_.ravel(),0]
        weights=len(y)/(2*np.bincount(y))[y]
        def loss(beta):
            eta=design@beta
            return np.sum(weights*(np.logaddexp(0,eta)-y*eta))+.5*np.sum(beta**2)
        epsilon=1e-4;directions=np.eye(3)*epsilon
        gradient=np.array([(loss(theta+d)-loss(theta-d))/(2*epsilon) for d in directions])
        hessian=np.array([[(loss(theta+a+b)-loss(theta+a-b)-loss(theta-a+b)+loss(theta-a-b))/(4*epsilon**2)
                          for b in directions] for a in directions])
        u=-gradient[2]+hessian[2,:2]@np.linalg.solve(hessian[:2,:2],gradient[:2])
        v=hessian[2,2]-hessian[2,:2]@np.linalg.solve(hessian[:2,:2],hessian[:2,2])
        self.assertAlmostEqual(gains[1],u*u/v,places=5)


if __name__=="__main__":
    unittest.main()
