"""Scientific invariants; igraph tests use the existing graph-worker environment."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from scripts import bernoulli_clustering_worker as worker
from scripts import run_bernoulli_clustering_analysis as study


class ClusteringTests(unittest.TestCase):
    def test_canonical_and_baseline_id_order(self):
        np.testing.assert_array_equal(worker.canonical([5, 2, 5, 8]), [0, 1, 0, 2])
        baseline = np.array([2, 2, 0, 1, 0])
        np.testing.assert_array_equal(worker.baseline_order([7, 7, 4, 8, 4], baseline), baseline)
        self.assertEqual(worker.changed_memberships(np.array([0, 0, 1]), np.array([1, 1, 0])), 0)
        self.assertEqual(worker.changed_memberships(np.array([0, 0, 1]), np.array([0, 0, 0])), 3)

    def test_hierarchy_chaining_missing_edges_and_isolates(self):
        baseline = np.arange(4)
        left, right, weight = np.array([0, 1]), np.array([1, 2]), np.array([.9, .9])
        complete = worker.hierarchy_labels(baseline, left, right, weight, 'complete', .8)
        average = worker.hierarchy_labels(baseline, left, right, weight, 'average', .4)
        self.assertNotEqual(complete[0], complete[2])
        self.assertEqual(average[0], average[2])
        self.assertNotEqual(average[2], average[3])
        np.testing.assert_array_equal(worker.hierarchy_labels(baseline, left, right, weight, 'average', .95), baseline)

    def test_hierarchy_uses_max_edge_and_never_splits(self):
        baseline = np.array([1, 1, 0, 0, 2])
        labels = worker.hierarchy_labels(baseline, np.array([0, 0]), np.array([1, 1]), np.array([.7, .95]), 'complete', .9)
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[4])
        self.assertEqual(labels[0], labels[1])
        with self.assertRaises(ValueError):
            worker.hierarchy_labels(baseline, np.array([]), np.array([]), np.array([]), 'ward', .8)

    def test_pair_count_weighting_and_sqrt_once(self):
        raw = sparse.csr_matrix([[1., 3., 5.], [4., 0., 2.]])
        train, test, sizes = study.aggregate_pair_clusters(raw, raw, np.array([0, 0, 0]))
        np.testing.assert_allclose(train.toarray().ravel(), np.sqrt([3., 2.]))
        np.testing.assert_allclose(test.toarray(), train.toarray())
        np.testing.assert_array_equal(sizes, [3])
        self.assertFalse(np.allclose(train.toarray().ravel(), np.sqrt(raw.toarray()).mean(axis=1)))

    def test_nearest_correlations_exclude_self_and_keep_actual_sign(self):
        x = sparse.csr_matrix(np.column_stack(([0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1])))
        result, matrix, neighbors = study.nearest_diagnostics(x)
        self.assertAlmostEqual(result['nearest_abs_median'], 1.)
        self.assertAlmostEqual(result['nearest_signed_median'], -1.)
        self.assertEqual(result['high_correlation_pairs'], 1)
        self.assertTrue(np.isnan(matrix[2]).all())
        self.assertTrue(all(n['feature_position'] != n['neighbor_position'] for n in neighbors))
        result, _, _ = study.nearest_diagnostics(x[:, :1])
        self.assertTrue(np.isnan(result['nearest_abs_median']))
        result, _, _ = study.nearest_diagnostics(x[:, :0])
        self.assertTrue(np.isnan(result['nearest_abs_max']))

    def test_cv_requires_all_folds_and_has_deterministic_ties(self):
        rows = [dict(method=s['method'], setting=s['setting'], requested_k=k, fold=f, auroc=.7, effective_k=k)
                for s in study.settings() for k in study.K_VALUES for f in range(3)]
        cv = pd.DataFrame(rows).sample(frac=1, random_state=4)
        _, locked = study.choose_settings(cv)
        self.assertTrue(locked.requested_k.eq(1).all())
        self.assertEqual(locked.set_index('method').loc['lower_resolution', 'setting'], 'resolution_0.75')
        self.assertEqual(locked.set_index('method').loc['hierarchy', 'setting'], 'average_cut_baseline')
        with self.assertRaises(ValueError):
            study.choose_settings(cv[cv.fold.ne(2)])

    def test_fold_isolation_and_exact_split(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.savez(root/'split.npz', train_index=[0, 1], test_index=[2], feature_ids=[0, 3])
            np.testing.assert_array_equal(study.verify_split(root, [0, 1], [2], [3]), [0, 3])
            with self.assertRaises(ValueError):
                study.verify_split(root, [0, 1], [2], [2])
            with self.assertRaises(ValueError):
                study.verify_split(root, [1, 0], [2], [3])

    def test_fresh_bh_over_whole_cluster_family(self):
        raw = sparse.csr_matrix(np.arange(12).reshape(4, 3))
        captured = []
        original = study.screen
        def screen(*args):
            captured.append(args[0].shape[1])
            return original(*args)
        with tempfile.TemporaryDirectory() as name, patch.object(study, 'screen', side_effect=screen):
            train, test, _ = study.screen_cached(raw, raw, np.array([0, 0, 1, 1]), np.array([0, 1, 2]), Path(name))
        self.assertEqual(captured, [3])
        self.assertEqual(train.shape[1], test.shape[1])

    def test_empty_and_exhausted_selections(self):
        target = np.array([0, 0, 1, 1])
        metric, _ = study.evaluate(sparse.csr_matrix((4, 0)), sparse.csr_matrix((4, 0)), target, target, [1, 10])
        self.assertTrue(metric.effective_k.eq(0).all())
        self.assertTrue(metric.auroc.eq(.5).all())
        one = sparse.csr_matrix([[0.], [1.], [2.], [3.]])
        metric, _ = study.evaluate(one, one, target, target, [1, 10])
        self.assertTrue(metric.effective_k.eq(1).all())

    def test_extended_leiden_in_actual_worker_environment(self):
        code = '''import json
import igraph as ig
from scripts.bernoulli_clustering_worker import extended_partition
g=ig.Graph.Full(5); g.es['weight']=[.9]*g.ecount()
a,b,trace,ok=extended_partition(g,.5,0)
a2,b2,trace2,ok2=extended_partition(g,.5,0)
assert trace == trace2 and b.tolist() == b2.tolist()
assert ok and ok2 and trace[-1]['extra_iterations'] >= 100
assert trace[-1]['stable_batches'] >= 3
assert all(row['gain'] >= -1e-10 for row in trace)
_,_,_,capped=extended_partition(g,.5,0,minimum=25,maximum=25,batch=25,patience=3)
assert not capped
print(json.dumps({'iterations':trace[-1]['extra_iterations'],'converged':ok}))
'''
        result = subprocess.run([str(study.JOIN_PYTHON), '-c', code], cwd=study.PROJECT, text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(result.stdout)['converged'])

    def test_full_graph_worker_resume_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            graph, output = root/'graph', root/'analysis'
            (graph/'edges').mkdir(parents=True)
            ids = np.array([10, 20, 30, 40, 50, 60, 70])
            baseline = np.array([1, 1, 0, 0, 2, 3, 4])
            np.savez(root/'baseline.npz', feature_ids=ids, clusters=baseline)
            edges = np.array([(10, 20, .9), (30, 40, -.9), (20, 30, .8), (50, 60, .99)],
                dtype=[('feature_j', '<u4'), ('feature_k', '<u4'), ('phi', '<f4')])
            np.save(graph/'edges/part-000000.npy', edges)
            (graph/'run.json').write_text(json.dumps(dict(status='complete', threshold=.8, edge_count=4,
                edge_weight=dict(field='phi', transform='abs'))))
            request = dict(graph=str(graph), output=str(output), baseline=str(root/'baseline.npz'), signature='fixture',
                hierarchy=[dict(key='complete_cut', linkage='complete', cutoff=.8)])
            (root/'request.json').write_text(json.dumps(request))
            protected = [(p, p.stat().st_size, p.stat().st_mtime_ns) for p in graph.rglob('*') if p.is_file()]
            command = [str(study.JOIN_PYTHON), str(study.WORKER), str(root/'request.json')]
            subprocess.run(command, cwd=study.PROJECT, text=True, capture_output=True, check=True)
            marker = output/'complete.json'
            stamp = marker.stat().st_mtime_ns
            subprocess.run(command, cwd=study.PROJECT, text=True, capture_output=True, check=True)
            self.assertEqual(marker.stat().st_mtime_ns, stamp)
            self.assertTrue(all(json.loads(marker.read_text())['converged'].values()))
            self.assertEqual(protected, [(p, p.stat().st_size, p.stat().st_mtime_ns) for p, _, _ in protected])
            with np.load(output/'resolution_1.npz') as saved:
                labels = saved['clusters']
                self.assertNotEqual(labels[0], labels[4])
                self.assertNotEqual(labels[4], labels[6])
            self.assertFalse((output/'.component-buckets').exists())

    def test_end_to_end_dataset_evaluation_with_mocked_clustering(self):
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
        from research_utils import normalize_rows, prevalence_mask
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root/'outputs/preprocessing/toy'
            source.mkdir(parents=True)
            y = np.repeat([0, 1], 20)
            raw = np.random.default_rng(42).uniform(.1, 1, size=(40, 6))
            raw[:, 0] += 5*y
            matrix = sparse.csr_matrix(raw)
            pb, pg = np.array([0, 0, 1, 1, 2, 2]), np.arange(6)
            sparse.save_npz(source/'X_valid.npz', matrix)
            pd.DataFrame(dict(sample_id=np.arange(40), y=y)).to_csv(source/'samples.csv', index=False)
            np.savez(source/'edges.npz', pair_bacteria=pb, pair_gene=pg)
            cached = root/'cached'
            outer_train, outer_test = next(StratifiedShuffleSplit(1, test_size=.2, random_state=42).split(matrix, y))
            folds = [(f'fold_{f}', outer_train[a], outer_train[b]) for f, (a, b) in enumerate(
                StratifiedKFold(3, shuffle=True, random_state=42).split(outer_train, y[outer_train]))]
            outer_graph = root/'outputs/method_comparison/real/toy/bernoulli_join_graph'
            for space, a, b in [*folds, ('outer', outer_train, outer_test)]:
                directory = cached/space
                directory.mkdir(parents=True)
                mask, _, _ = prevalence_mask(matrix[a], .05, 5)
                ids, labels = np.flatnonzero(mask), np.arange(mask.sum())
                np.savez(directory/'split.npz', train_index=a, test_index=b, feature_ids=ids)
                graph = outer_graph if space == 'outer' else directory/'join'
                graph.mkdir(parents=True)
                np.savez(graph/'assignments.npz', feature_ids=ids, clusters=labels)
                if space == 'outer':
                    np.savez(graph.parent/'bernoulli_join_graph_split.npz', train_index=a, test_index=b, feature_ids=ids)
                train, test = normalize_rows(matrix[a][:, ids]), normalize_rows(matrix[b][:, ids])
                jt, jv, sizes = study.aggregate_pair_clusters(train, test, labels)
                bt, bids = study.collapsed_average(train, pb[ids], 3)
                bv, _ = study.collapsed_average(test, pb[ids], 3)
                frames = []
                for family, t, v, fids, counts in [('join', jt, jv, labels, sizes), ('bacteria', bt, bv, bids, np.bincount(pb[ids])[bids])]:
                    t, v, table = study.screen(t, v, y[a], fids, family, counts)
                    sparse.save_npz(directory/f'{family}_train.npz', t)
                    sparse.save_npz(directory/f'{family}_test.npz', v)
                    table.to_csv(directory/f'{family}_features.csv.gz', index=False)
                    if space == 'outer':
                        metric, _ = study.evaluate(t, v, y[a], y[b], study.K_VALUES)
                        frames.append(metric.assign(method='baseline' if family == 'join' else family))
                if space == 'outer':
                    pd.concat(frames).to_csv(cached/'metrics.csv', index=False)
            output = root/'analysis'
            output.mkdir()
            (output/'protocol.json').write_text(json.dumps(dict(signature='fixture')))
            def clustering(dataset, space, graph, target):
                target.mkdir(parents=True)
                with np.load(graph/'assignments.npz') as saved:
                    ids, labels = saved['feature_ids'], saved['clusters']
                for setting in [*study.settings(), dict(setting='fresh_control')]:
                    np.savez(target/f'{setting["setting"]}.npz', feature_ids=ids, clusters=labels)
                rows = [dict(component=0, resolution=r, component_weight=1., extra_iterations=i,
                    quality=.5, membership_changes=0, converged=True) for r in worker.RESOLUTIONS for i in [0, 25, 50, 75, 100]]
                pd.DataFrame(rows).to_csv(target/'convergence.csv.gz', index=False)
                return dict(effective_threshold=.8, clusters=len(labels))
            with patch.object(study, 'PROJECT', root), patch.object(study, 'OUTPUT', output), \
                 patch.object(study, 'cached_root', return_value=cached), patch.object(study, 'clustering', side_effect=clustering):
                study.dataset_run('toy')
                study.DATASETS, original_datasets = ['toy'], study.DATASETS
                try:
                    study.summarize()
                finally:
                    study.DATASETS = original_datasets
            self.assertEqual(json.loads((output/'toy/result.json').read_text())['status'], 'complete')
            self.assertTrue((output/'toy/fine_k_metrics.png').exists())
            self.assertTrue((output/'coarse_k_aggregate_real.png').exists())
            locked = pd.read_csv(output/'toy/locked_parameters.csv')
            self.assertEqual(set(locked.method), set(study.METHODS))


if __name__ == '__main__':
    unittest.main()
