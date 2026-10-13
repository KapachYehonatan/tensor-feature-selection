"""Scientific invariants for test-free selection, using small in-memory data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from scripts import run_test_free_feature_selection as study


class TestFreeSelectionTests(unittest.TestCase):
    def test_resume_monitor_records_signal_exit_and_peak_without_research_changes(self):
        import subprocess
        import sys
        from scripts import resume_test_free_feature_selection as resume
        with tempfile.TemporaryDirectory() as tmp, patch.object(resume, 'SAMPLE_SECONDS', .01):
            child = subprocess.Popen([sys.executable, '-c',
                'import os,signal,time; data=bytearray(5000000); time.sleep(.1); os.kill(os.getpid(),signal.SIGTERM)'])
            record = resume.watch(child, Path(tmp)/'resources.csv')
            samples = pd.read_csv(Path(tmp)/'resources.csv')
        self.assertEqual(record['return_code'], -15)
        self.assertEqual(record['termination_signal'], 15)
        self.assertGreater(record['peak_rss_kib'], 0)
        self.assertGreater(record['peak_sampled_rss_kib'], 0)
        self.assertFalse(samples.empty)

    def test_zero_aware_states_use_only_positive_training_median(self):
        values = np.array([[0, 0, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0]], float)
        states, median = study.three_states(values)
        np.testing.assert_array_equal(median, [2, 2, 0])
        np.testing.assert_array_equal(states, [[0, 0, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0]])
        with self.assertRaises(ValueError): study.three_states(np.array([[-1.]]))

    def test_mi_is_test_free_and_zero_scores_do_not_fill_k(self):
        x = sparse.csr_matrix([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        y = np.array([0, 0, 1, 1])
        with (patch('scripts.run_method_comparison.bh_family', side_effect=AssertionError('No BH')),
              patch('scripts.run_posthoc_penalty_search.mw_effect', side_effect=AssertionError('No MW'))):
            score = study.mi_scores(x, y)
        self.assertAlmostEqual(score[0], np.log(2))
        np.testing.assert_allclose(score[1:], 0, atol=1e-12)
        np.testing.assert_array_equal(study.rank_scores(score, [10, 11, 12]), [0])
        np.testing.assert_array_equal(study.rank_scores([.5, .5, 0], [20, 10, 0]), [1, 0])
        self.assertEqual(len(study.rank_scores([], [])), 0)
        with self.assertRaises(ValueError): study.rank_scores([np.nan], [0])
        with self.assertRaises(ValueError): study.rank_scores([1, 2], [0, 0])

    def test_logistic_selectors_fit_standardized_continuous_inputs_without_tests(self):
        x = sparse.csr_matrix([[0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        scaler = StandardScaler(with_mean=False).fit(x)
        before = scaler.scale_.copy()
        scaler.transform(sparse.csr_matrix([[100., 100.]]))
        np.testing.assert_array_equal(before, scaler.scale_)
        with patch('scripts.run_method_comparison.bh_family', side_effect=AssertionError('No BH')):
            for method in ('l2', 'l1', 'elastic_net', 'mi_l1'):
                a, meta = study.logistic_scores(scaler.transform(x), y, method, .1)
                b, _ = study.logistic_scores(scaler.transform(x), y, method, .1)
                self.assertEqual(meta['status'], 'complete')
                np.testing.assert_array_equal(a, b)
        coef, meta = study.logistic_scores(x[:, :0], y, 'l1', 1)
        self.assertEqual(len(coef), 0)
        self.assertEqual(meta['attempts'], 0)

    def test_retry_then_failure_never_substitutes_selector(self):
        x = sparse.csr_matrix([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        def estimator(fails):
            model = Mock(coef_=np.array([[.3]]), n_iter_=np.array([1]), solver='liblinear')
            def fit(*args):
                if fails: warnings.warn('not converged', ConvergenceWarning)
                return model
            model.fit.side_effect = fit
            return model
        with patch.object(study, 'LogisticRegression', side_effect=[estimator(True), estimator(False)]) as factory:
            _, meta = study.logistic_scores(x, y, 'l1', 1)
            self.assertEqual(meta['attempts'], 2)
            self.assertEqual([c.kwargs['max_iter'] for c in factory.call_args_list], list(study.FIT_LIMITS))
        with patch.object(study, 'LogisticRegression', side_effect=[estimator(True), estimator(True)]):
            with self.assertRaises(RuntimeError): study.logistic_scores(x, y, 'elastic_net', 1)

    def test_shortlisting_is_confined_to_named_method_and_constants_are_excluded(self):
        x = sparse.csr_matrix([[0, 0, 4], [1, 0, 4], [2, 1, 4], [3, 1, 4]], dtype=float)
        y = np.array([0, 0, 1, 1])
        widths = []
        def coefficients(scaled, target, method, c):
            widths.append((method, scaled.shape[1]))
            return np.ones(scaled.shape[1]), dict(status='complete')
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, 'SHORTLIST', 1), \
             patch.object(study, 'mi_scores', return_value=np.array([.1, .5])), \
             patch.object(study, 'logistic_scores', side_effect=coefficients):
            rank = study.fit_rankings(x, y, np.ones(3), Path(tmp), {'l1': [1], 'mi_l1': [1]})
        self.assertEqual(widths, [('l1', 2), ('mi_l1', 1)])
        np.testing.assert_array_equal(rank[('l1', 1)], [0, 1])
        np.testing.assert_array_equal(rank[('mi_l1', 1)], [1])

    def test_cv_choice_is_independent_per_method_with_no_reference_fallback(self):
        rows = []
        for method, cs in {'baseline': [0.], 'bacteria': [0.], **study.GRID}.items():
            for c in cs:
                for k in study.K_VALUES:
                    for fold in range(3):
                        auc = .99 if method in ('baseline', 'bacteria') else .6
                        if method == 'l1' and c in (.01, .1) and k in (2, 3): auc = .7
                        rows.append(dict(method=method, parameter=c, requested_k=k, effective_k=k, fold=fold, auroc=auc))
        cv = pd.DataFrame(rows)
        _, locked = study.choose_settings(cv)
        chosen = locked.set_index('method')
        self.assertEqual((chosen.loc['l1', 'parameter'], chosen.loc['l1', 'requested_k']), (.01, 2))
        self.assertEqual(set(chosen.index), set(study.METHODS))
        pd.testing.assert_frame_equal(study.choose_settings(cv.sample(frac=1, random_state=3))[1], locked)
        with self.assertRaises(ValueError): study.choose_settings(cv.iloc[1:])

    def test_fold_isolation_and_weighted_mean_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = Path(tmp)/'fold_0'
            space.mkdir()
            np.savez(space/'split.npz', train_index=[0, 1], test_index=[2], feature_ids=[0, 1])
            study.verify_split(space, np.array([0, 1, 2]), np.array([3]), 4)
            with self.assertRaises(ValueError): study.verify_split(space, np.array([0, 1]), np.array([2, 3]), 4)
            raw = sparse.csr_matrix([[.1, .5, .4], [.2, .2, .6]])
            train, test, _ = study.aggregate_pair_clusters(raw, raw, [0, 0, 1])
            np.testing.assert_allclose(train.toarray(), np.sqrt([[.3, .4], [.2, .6]]))
            for suffix in ('train', 'test'): sparse.save_npz(space/f'join_{suffix}.npz', train[:, [1]])
            pd.DataFrame(dict(feature_id=[1])).to_csv(space/'join_features.csv.gz', index=False)
            study.verify_predictors(train, test, space)
            with self.assertRaises(ValueError): study.verify_predictors(train*2, test, space)


if __name__ == '__main__': unittest.main()
