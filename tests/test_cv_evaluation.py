"""Fold isolation, full-pipeline toy run, OOF metrics and deterministic resumption."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from scripts import run_cv_evaluation as study


class CVTests(unittest.TestCase):
    def test_folds_cover_every_sample_once_and_keep_stratification(self):
        y = np.r_[np.zeros(27, int), np.ones(103, int)]
        folds = study.make_folds(y)
        self.assertEqual(len(folds), 5)
        self.assertEqual(sorted(np.concatenate([b for a, b in folds]).tolist()), list(range(len(y))))
        self.assertEqual([len(b) for a, b in folds], [26]*5)
        self.assertEqual(sorted(np.bincount(y[b])[0] for a, b in folds), [5, 5, 5, 6, 6])
        for (a, b), (c, d) in zip(folds, study.make_folds(y)):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
            self.assertEqual(len(np.intersect1d(a, b)), 0)
        with self.assertRaises(ValueError): study.make_folds(np.r_[np.zeros(4, int), np.ones(10, int)])

    def test_validation_only_features_and_values_do_not_enter_prevalence(self):
        x = sparse.csr_matrix(np.column_stack((np.ones(20), np.r_[np.zeros(10), np.ones(10)])))
        ids, cutoff = study.training_pairs(x, np.arange(10))
        np.testing.assert_array_equal(ids, [0])
        self.assertEqual(cutoff, 5)
        altered = x.toarray()
        altered[10:, :] *= 1000
        np.testing.assert_array_equal(study.training_pairs(sparse.csr_matrix(altered), np.arange(10))[0], ids)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'split.npz'
            study.verify_split(path, np.arange(10), np.arange(10, 20), ids, cutoff)
            with self.assertRaises(ValueError): study.verify_split(path, np.arange(9), np.arange(9, 20), ids, cutoff)

    def test_validation_labels_cannot_change_rankings_or_training_scaling(self):
        from feature_selection.run_feature_selection import new_model
        y = np.repeat([0, 1], 15)
        values = np.column_stack((np.ones(30), 1+10*y, np.linspace(.1, 1, 30)))
        x = sparse.csr_matrix(values)
        a, b = np.r_[np.arange(10), np.arange(15, 25)], np.r_[np.arange(10, 15), np.arange(25, 30)]
        pb, pg, ids, labels = np.array([0, 0, 1]), np.array([0, 1, 1]), np.arange(3), np.array([0, 1, 2])
        changed = y.copy()
        changed[b] = 1-changed[b]
        with tempfile.TemporaryDirectory() as name:
            first, second = Path(name)/'first', Path(name)/'second'
            first.mkdir(); second.mkdir()
            captured = []
            original = study.screen
            def screen(*args):
                captured.append((args[0].shape[1], args[2].copy()))
                return original(*args)
            with patch.object(study, 'screen', side_effect=screen):
                study.prepare_features(x, y, pb, pg, a, b, ids, labels, first)
                study.prepare_features(x, changed, pb, pg, a, b, ids, labels, second)
            self.assertEqual([n for n, target in captured[:3]], [2, 2, 3])
            for n, target in captured: np.testing.assert_array_equal(target, y[a])
            for method in study.METHODS:
                t1 = pd.read_csv(first/f'{method}_features.csv.gz')
                t2 = pd.read_csv(second/f'{method}_features.csv.gz')
                pd.testing.assert_frame_equal(t1, t2)
        model = new_model().fit(x[a], y[a])
        before = model.named_steps['standardscaler'].scale_.copy()
        model.predict_proba(x[b]*1000)
        np.testing.assert_array_equal(model.named_steps['standardscaler'].scale_, before)

    def test_mean_and_pooled_auc_are_distinct_and_match_saved_predictions(self):
        pred = pd.DataFrame(dict(method=['join']*4, requested_k=[1]*4, fold=[0, 0, 1, 1],
            y=[0, 1, 0, 1], probability=[.1, .2, .8, .9]))
        mean = np.mean([study.prediction_metrics(g).auroc.iloc[0] for f, g in pred.groupby('fold')])
        pooled = study.prediction_metrics(pred).auroc.iloc[0]
        self.assertEqual(mean, 1.)
        self.assertEqual(pooled, .75)
        self.assertEqual(pooled, roc_auc_score(pred.y, pred.probability))

    def test_weighting_sqrt_constants_empty_and_exhausted_candidates(self):
        raw = sparse.csr_matrix([[1., 3., 5.], [4., 0., 2.]])
        train, test, sizes = study.aggregate_pair_clusters(raw, raw, [0, 0, 0])
        np.testing.assert_allclose(train.toarray().ravel(), np.sqrt([3., 2.]))
        marginal, ids = study.collapsed_average(raw, [0, 0, 1], 2)
        np.testing.assert_allclose(marginal.toarray(), np.sqrt([[2., 5.], [2., 2.]]))
        y = np.repeat([0, 1], 6)
        x = sparse.csr_matrix(np.column_stack((np.ones(12), y, y)))
        t, v, table = study.screen(x, x, y, np.arange(3), 'pair_cluster', [1, 2, 3])
        self.assertEqual(table.feature_id.tolist(), [1, 2])
        metric, _ = study.evaluate(t, v, y, y, [1, 100])
        self.assertEqual(metric.effective_k.tolist(), [1, 2])
        metric, pred = study.evaluate(x[:, :0], x[:, :0], y, y, [1, 100])
        self.assertTrue(metric.effective_k.eq(0).all())
        self.assertTrue(metric.auroc.eq(.5).all())
        self.assertTrue(pred.probability.eq(y.mean()).all())

    def test_real_exact_join_toy_five_fold_run_and_resume(self):
        rng = np.random.default_rng(42)
        y = np.repeat([0, 1], 20)
        detections = rng.binomial(1, .5, size=(40, 2))
        raw = np.column_stack((np.ones(40), 1+8*y, detections[:, 0], detections[:, 0], detections[:, 1]))
        x = sparse.csr_matrix(raw)
        samples = pd.DataFrame(dict(sample_id=[f's{i}' for i in range(40)], y=y))
        pb, pg = np.array([0, 1, 2, 2, 3]), np.array([0, 0, 1, 2, 2])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            matrix_path = root/'raw.npz'
            sparse.save_npz(matrix_path, x)
            with (patch.object(study, 'OUTPUT', root/'output'), patch.object(study, 'THRESHOLDS', {'toy': .7}),
                  patch.object(study, 'JOIN_WORKERS', 2), patch.object(study, 'COMMUNITY_WORKERS', 1),
                  patch.object(study, 'WORKER_MEMORY_GIB', 1), patch.object(study, 'JOIN_WORKING_GIB', 1),
                  patch.object(study, 'COMMUNITY_WORKING_GIB', 1), patch.object(study, 'RESERVE_GIB', 0),
                  patch.object(study, 'FREE_DISK_GIB', 0), patch.object(study, 'OUTPUT_GIB', 1),
                  patch.object(study, 'KEEP_GRAPH_GIB', 1), patch.object(study, 'plot_metrics')):
                for fold, (a, b) in enumerate(study.make_folds(y)):
                    study.fold_run('toy', fold, x, samples, pb, pg, a, b, matrix_path, 'fixture')
                    directory = study.OUTPUT/'toy'/f'fold_{fold}'
                    with np.load(directory/'split.npz') as split:
                        np.testing.assert_array_equal(split['train_index'], a)
                        np.testing.assert_array_equal(split['test_index'], b)
                    meta = json.loads((directory/'join_graph/packed/metadata.json').read_text())
                    self.assertEqual(meta['n'], len(a))
                    with np.load(directory/'split.npz') as split:
                        ids = split['feature_ids']
                    words = np.load(directory/'join_graph/packed/bits.npy')
                    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder='little')[:, :len(a)]
                    np.testing.assert_array_equal(bits, x[a][:, ids].toarray().T > 0)
                    integration = json.loads((directory/'join_graph/integration.json').read_text())
                    self.assertEqual(integration['community']['workers'], 1)
                mean, sd, pooled = study.summarize_dataset('toy', samples)
                self.assertEqual(len(mean), len(study.METHODS)*len(study.K_VALUES))
                prediction_path = study.OUTPUT/'toy/oof_predictions.csv.gz'
                before = prediction_path.read_bytes()
                original_predictions = pd.read_csv(prediction_path, float_precision='round_trip')
                timestamps = {p: p.stat().st_mtime_ns for p in (study.OUTPUT/'toy').glob('fold_*/complete.json')}
                with patch.object(study, 'cluster_fold', side_effect=AssertionError('Completed graphs must not rerun')):
                    for fold, (a, b) in enumerate(study.make_folds(y)):
                        study.fold_run('toy', fold, x, samples, pb, pg, a, b, matrix_path, 'fixture')
                self.assertEqual(timestamps, {p: p.stat().st_mtime_ns for p in timestamps})
                self.assertEqual(before, prediction_path.read_bytes())
                study.summarize_dataset('toy', samples)
                after = pd.read_csv(prediction_path, float_precision='round_trip')
                pd.testing.assert_frame_equal(original_predictions, after)
                self.assertEqual(len(after), 40*len(study.METHODS)*len(study.K_VALUES))
                self.assertFalse(after.duplicated(['method', 'requested_k', 'sample_index']).any())
                # Broken coverage must not become a deceptively complete aggregate.
                frame = pd.read_csv(study.OUTPUT/'toy/fold_0/predictions.csv.gz')
                frame.iloc[1:].to_csv(study.OUTPUT/'toy/fold_0/predictions.csv.gz', index=False)
                with self.assertRaises(ValueError): study.summarize_dataset('toy', samples)


if __name__ == '__main__': unittest.main()
