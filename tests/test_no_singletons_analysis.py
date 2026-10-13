"""Singleton removal precedes statistical screening; folds/prediction stay unchanged."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from scripts import run_no_singletons_analysis as study


class NoSingletonTests(unittest.TestCase):
    def test_remove_before_testing_full_remaining_family_with_original_ids(self):
        x = sparse.csr_matrix(np.arange(24).reshape(6, 4))
        captured = []
        original = study.screen
        def screen(*args):
            captured.append((args[0].toarray(), args[3], args[5]))
            return original(*args)
        with patch.object(study, 'screen', side_effect=screen):
            t, v, table = study.screen_non_singletons(x, x, np.repeat([0, 1], 3), np.array([1, 3, 1, 2]))
        np.testing.assert_array_equal(captured[0][0], x[:, [1, 3]].toarray())
        np.testing.assert_array_equal(captured[0][1], [1, 3])
        np.testing.assert_array_equal(captured[0][2], [3, 2])
        self.assertTrue(table.feature_id.isin([1, 3]).all())
        self.assertTrue(table.pair_count.gt(1).all())
        self.assertEqual(t.shape, v.shape)

    def test_empty_family_constants_and_exhaustion(self):
        y = np.repeat([0, 1], 4)
        x = sparse.csr_matrix(np.column_stack((np.ones(8), y)))
        for sizes in (np.array([1, 1]), np.array([2, 1])):
            t, v, table = study.screen_non_singletons(x, x, y, sizes)
            self.assertEqual(len(table), 0)
            metrics, _ = study.evaluate(t, v, y, y, [1, 100])
            self.assertTrue(metrics.effective_k.eq(0).all())
            self.assertTrue(metrics.auroc.eq(.5).all())
        t, v, table = study.screen_non_singletons(x, x, y, np.array([1, 2]))
        self.assertEqual(table.feature_id.tolist(), [1])
        metric, _ = study.evaluate(t, v, y, y, [1, 100])
        self.assertTrue(metric.effective_k.eq(1).all())

    def test_invalid_counts_and_deterministic_selection(self):
        x = sparse.csr_matrix(np.column_stack((np.repeat([0, 1], 10), np.repeat([0, 1], 10))))
        y = np.repeat([0, 1], 10)
        for sizes in (np.array([0, 2]), np.array([2]), np.array([2., 3.])):
            with self.assertRaises(ValueError): study.screen_non_singletons(x, x, y, sizes)
        _, _, table = study.screen_non_singletons(x, x, y, np.array([2, 3]))
        self.assertEqual(table.feature_id.tolist(), [0, 1])
        _, _, again = study.screen_non_singletons(x, x, y, np.array([2, 3]))
        pd.testing.assert_frame_equal(table, again)

    def test_k_selection_complete_folds_and_ties(self):
        cv = pd.DataFrame([dict(method=m, fold=f, requested_k=k, auroc=.7)
            for m in study.METHODS for f in range(3) for k in study.K_VALUES])
        _, locked = study.choose_k(cv.sample(frac=1, random_state=9))
        self.assertTrue(locked.requested_k.eq(1).all())
        with self.assertRaises(ValueError): study.choose_k(cv[cv.fold.ne(2)])
        with self.assertRaises(ValueError): study.choose_k(pd.concat([cv, cv.iloc[:1]]))


if __name__ == '__main__': unittest.main()
