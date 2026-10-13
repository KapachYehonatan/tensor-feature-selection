"""Shared scientific invariants and explicit refactor-cache compatibility."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import research_utils as u


class SharedTests(unittest.TestCase):
    def test_aggregate_plot_does_not_hide_standard_deviation(self):
        import pandas as pd
        from scripts import summarize_method_comparison as summary
        rows = [dict(method=m, requested_k=k, **{metric:value for metric in summary.METRICS})
                for m in summary.METHOD_ORDER for k, value in [(1, .7), (2, .8)]]
        curves = pd.DataFrame(rows)
        errors = curves.copy()
        errors[summary.METRICS] = .2
        with tempfile.TemporaryDirectory() as name, patch.object(summary, 'FIGURES', Path(name)):
            with patch.object(summary.plt, 'close'):
                summary.plot_curves(curves, 'fixture', 'fixture', errors)
                figure = summary.plt.gcf()
                for axis in figure.axes:
                    self.assertLessEqual(axis.get_ylim()[0], .5)
                    self.assertGreaterEqual(axis.get_ylim()[1], 1.)
            summary.plt.close(figure)

    def test_fresh_three_method_holdout_and_cached_resumption(self):
        from scripts import run_method_comparison as comparison
        from scripts import run_bernoulli_correlation_join as join
        import pandas as pd
        rng = np.random.default_rng(42)
        y = np.repeat([0, 1], 20)
        detection = rng.binomial(1, .5, 40)
        matrix = sparse.csr_matrix(np.column_stack((np.ones(40), 1+8*y, detection, detection)))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sparse.save_npz(root/'X_valid.npz', matrix)
            np.savez(root/'edges.npz', pair_bacteria=[0, 1, 2, 2], pair_gene=[0, 0, 1, 2])
            pd.DataFrame(dict(sample_id=[f's{i}' for i in range(40)], y=y)).to_csv(root/'samples.csv', index=False)
            job = dict(source='real', dataset='toy', n=None, seed=None, scenario='real', input=str(root))
            with (patch.object(comparison, 'OUTPUT_ROOT', root/'output'), patch.object(join, 'OUTPUT_ROOT', root/'output'),
                  patch.object(join, 'WORKERS', 1), patch.object(join, 'MEMORY_GIB', 1),
                  patch.object(join, 'WORKING_MEMORY_GIB', 1), patch.object(join, 'COMMUNITY_WORKING_MEMORY_GIB', 1),
                  patch.object(join, 'MEMORY_RESERVE_GIB', 0), patch.object(join, 'FREE_DISK_GIB', 0),
                  patch.object(join, 'OUTPUT_GIB', 1), patch.object(join, 'KEEP_GRAPH_GIB', 1)):
                comparison.run_job(job)
                join.run_dataset(job)
                for method in comparison.METHODS:
                    table = pd.read_csv(comparison.output_path(job)/method/'metrics.csv')
                    self.assertEqual(table.requested_k.tolist(), comparison.K_VALUES)
                    self.assertTrue(table.auroc.between(0, 1).all())
                with (patch.object(comparison, 'load_job', side_effect=AssertionError('Cached baselines refit')),
                      patch.object(join, 'load_job', side_effect=AssertionError('Cached graph rebuilt'))):
                    comparison.run_job(job)
                    join.run_dataset(job)
                with patch.object(comparison, 'MODEL_SEED', 99), self.assertRaisesRegex(ValueError, 'inputs/code changed'):
                    comparison.run_job(job)

    def test_prevalence_normalization_and_training_denominators(self):
        x = sparse.csr_matrix([[1., 3., 8.], [2., 2., 0.], [0., 4., 4.]])
        mask, support, cutoff = u.prevalence_mask(x, .5, 1)
        np.testing.assert_array_equal(mask, [True, True, True])
        self.assertEqual(cutoff, 2)
        norm = u.normalize_rows(x)
        np.testing.assert_allclose(np.asarray(norm.sum(axis=1)).ravel(), 1.)
        collapsed, ids = u.collapsed_average(norm, [0, 0, 1], 2)
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(collapsed.toarray(), np.sqrt(
            np.column_stack(((norm.toarray()[:, 0]+norm.toarray()[:, 1])/2, norm.toarray()[:, 2]))))

    def test_bh_constants_and_deterministic_ties(self):
        np.testing.assert_allclose(u.benjamini_hochberg([.01, .04, .03, .8]), [.04, .053333333333, .053333333333, .8])
        y = np.repeat([0, 1], 10)
        x = sparse.csr_matrix(np.column_stack((np.ones(20), y, y)))
        scores = u.bh_family(x, [y])
        self.assertTrue(np.isnan(scores['p'][0]))
        self.assertEqual(scores['q'][0], 1.)
        self.assertEqual(scores['order'].tolist(), [1, 2, 0])
        t, v, table = u.screen(x, x, y, np.arange(3), 'pair_cluster', [1, 1, 1])
        self.assertEqual(table.feature_id.tolist(), [1, 2])
        metrics, prediction = u.evaluate(t, v, y, y, [1, 100])
        self.assertEqual(metrics.effective_k.tolist(), [1, 2])

    def test_empty_selection_uses_training_prevalence(self):
        y = np.r_[np.zeros(7), np.ones(3)]
        predictions = u.predict_model(sparse.csr_matrix((10, 0)), y, sparse.csr_matrix((4, 0)))
        np.testing.assert_allclose(predictions, .3)

    def test_participant_selection_is_independent_and_deterministic(self):
        mask = u.unique_day_zero_stool_mask(['a', 'a', 'b', 'b'], ['stool']*4,
            [0, 0, 0, 1], [10, 20, 30, 40], ['s1', 's2', 's3', 's4'])
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_legacy_cache_requires_explicit_exact_bridge_and_unchanged_inputs(self):
        with tempfile.TemporaryDirectory() as name, patch.object(u, 'PROJECT', Path(name)):
            root = Path(name)
            (root/'logs').mkdir()
            marker, source, code = root/'protocol.json', root/'data.bin', root/'core.py'
            marker.write_text(json.dumps(dict(signature='old')))
            source.write_bytes(b'input')
            code.write_text('original')
            self.assertEqual(u.cache_signature(marker, 'new'), 'new')
            bridge = dict(entries={'protocol.json':dict(old_signature='old', new_signature='new')},
                code_hashes={str(code):u.file_hash(code)},
                input_states=[[str(source), source.stat().st_size, source.stat().st_mtime_ns]],
                cache_states=[])
            (root/'logs/project_cleanup_compatibility.json').write_text(json.dumps(bridge))
            self.assertEqual(u.cache_signature(marker, 'new'), 'old')
            self.assertEqual(u.cache_signature(marker, 'unknown'), 'unknown')
            code.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'code changed'): u.cache_signature(marker, 'new')
            code.write_text('original')
            source.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): u.cache_signature(marker, 'new')


if __name__ == '__main__':
    unittest.main()
