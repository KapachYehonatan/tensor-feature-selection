"""Small supplied-value, inference, stage-cache, and reporting fixtures only."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock, patch
from types import SimpleNamespace
import fcntl
import itertools
import hashlib
import json
import tempfile
import unittest
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.io import mmwrite
from datasets.loading import load, validate
from datasets.prepare import prepare, prepare_matrix
from datasets import acquire
from execution import cache, pipeline, resources
from execution.spec import Experiment, Evaluation
from methods.preprocessing import Preprocessing, retained_pair_values
from methods.pipelines import Bacteria, Gene, BernoulliJoin, BinsEValue, construct
from methods.graphs.bernoulli_join import BernoulliGraph
from methods.clustering.leiden import Leiden
from methods.graphs import preparation as graph_preparation
from methods.clustering import partition
from methods.selectors import bins
from methods.selectors.mann_whitney import MannWhitney, mann_whitney_many, prepare_mann_whitney
from methods.selectors.correlation import greedy_filter
from methods.correction import BY, EBH
from metrics.prediction import predict_model, summarize
from metrics.correlation import average_pairwise_absolute_correlation
from metrics.shuffled_fdp import label_permutations, discovery_statistics
from reporting import comparison


@dataclass(frozen=True)
class TinyGraph(BernoulliGraph):
    calls: ClassVar[int] = 0

    def build(self, matrix, selection, output):
        TinyGraph.calls += 1
        output.mkdir()
        cache.write_json(output / "graph.json", {"fixture": True})


@dataclass(frozen=True)
class TinyLeiden(Leiden):
    calls: ClassVar[int] = 0

    def fit(self, graph, matrix, selection, output, graph_settings):
        TinyLeiden.calls += 1
        output.mkdir()
        with np.load(selection) as saved:
            ids = saved["feature_ids"]
        np.savez_compressed(output / "assignments.npz", feature_ids=ids, clusters=np.arange(len(ids)))
        return {"status": "complete"}


@dataclass(frozen=True)
class OtherLeiden(TinyLeiden):
    def fit(self, graph, matrix, selection, output, graph_settings):
        return super().fit(graph, matrix, selection, output, graph_settings)


@dataclass(frozen=True)
class FixedP(MannWhitney):
    calls: ClassVar[int] = 0

    def batch_test(self, matrix, targets, prepared=None):
        FixedP.calls += 1
        return np.full((len(targets), matrix.shape[1]), .05)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patches = []
        for module, name, value in ((cache, "PROJECT", self.root), (cache, "CACHE", self.root / "cache"),
                                    (comparison, "PROJECT", self.root)):
            p = patch.object(module, name, value)
            p.start()
            self.addCleanup(p.stop)
        TinyGraph.calls = TinyLeiden.calls = FixedP.calls = 0

    def dataset(self):
        directory = cache.CACHE / "inputs/toy"
        directory.mkdir(parents=True, exist_ok=True)
        y = np.arange(24) % 2
        values = np.column_stack((np.full(24, 2), y + 1, np.arange(24) % 3, np.r_[np.zeros(20), np.ones(4)]))
        sparse.save_npz(directory / "X_valid.npz", sparse.csr_matrix(values))
        np.savez_compressed(directory / "edges.npz", pair_bacteria=[0, 0, 1, 2], pair_gene=[0, 1, 0, 1])
        pd.DataFrame(dict(sample_id=[f"s{i}" for i in range(24)], subject_id=[f"p{i}" for i in range(24)], y=y)).to_csv(directory / "samples.csv", index=False)
        return load("toy")

    def test_supervisor_requires_memory_and_swap_caps(self):
        caps = {'memory.max': str(resources.LIMIT_BYTES), 'memory.swap.max': '0'}
        child = Mock(pid=123, returncode=0)
        child.poll.return_value = 0
        with patch.object(resources, 'PROJECT', self.root), \
             patch.object(resources.importlib.util, 'find_spec', return_value=SimpleNamespace(
                 origin=str(self.root / 'experiments/main_comparison.py'))), \
             patch.object(resources.sys, 'argv', ['resources', 'experiments.main_comparison']), \
             patch.object(resources.subprocess, 'Popen', return_value=child) as launch:
            for group in ({}, dict(caps, **{'memory.max': 'max'}),
                          dict(caps, **{'memory.max': str(resources.LIMIT_BYTES + 1)}),
                          {'memory.max': str(resources.LIMIT_BYTES)},
                          dict(caps, **{'memory.swap.max': 'max'}),
                          dict(caps, **{'memory.swap.max': '1'})):
                with self.subTest(group=group), patch.object(resources, 'cgroup_values', return_value=group):
                    with self.assertRaisesRegex(RuntimeError, 'MemorySwapMax=0'):
                        resources.main()
            launch.assert_not_called()
            with patch.object(resources, 'cgroup_values', return_value=caps), self.assertRaises(SystemExit) as stopped:
                resources.main()
            self.assertEqual(stopped.exception.code, 0)
            launch.assert_called_once()

    def test_supervisor_custom_logs_share_the_experiment_lock(self):
        logs = self.root / 'outputs/main_comparison/.cache/logs'
        logs.mkdir(parents=True)
        alternate = self.root / 'outputs/main_comparison/.cache/other_logs'
        with (logs / 'main_comparison.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(resources, 'PROJECT', self.root), \
                 patch.object(resources.importlib.util, 'find_spec', return_value=SimpleNamespace(
                     origin=str(self.root / 'experiments/main_comparison.py'))), \
                 patch.object(resources.sys, 'argv', ['resources', 'experiments.main_comparison',
                     '--log-directory', str(alternate)]), \
                 patch.object(resources.subprocess, 'Popen') as launch:
                with self.assertRaises(BlockingIOError):
                    resources.main()
                launch.assert_not_called()

    def test_common_preprocessing_supplied_means_and_validation_isolation(self):
        data = self.dataset()
        a, b = np.arange(20), np.arange(20, 24)
        ids, cutoff = Preprocessing().fit(data.values, a)
        np.testing.assert_array_equal(ids, [0, 1, 2])
        self.assertEqual(cutoff, 5)
        np.testing.assert_array_equal(retained_pair_values(data.values, ids).toarray(), data.values[:, ids].toarray())
        family, feature_ids, size = construct(data, a, ids, Bacteria())
        np.testing.assert_allclose(family[:, 0].toarray().ravel(), (3 + data.samples.y) / 2)
        self.assertEqual(family[0, 0], 1.5)  # Constant raw member remains in the mean.
        raw, raw_ids, raw_size = construct(data, a, ids, BinsEValue())
        np.testing.assert_array_equal(raw_ids, [1, 2])
        self.assertEqual(raw_size, 4)
        context = pipeline.Context("fold_0", a, b)
        before = pipeline.prepare_family(data, context, Preprocessing(), Bacteria())
        _, selected, _ = pipeline.selection(data, context, before, Bacteria(), 3)
        data.values[b] = data.values[b] * 17
        data.samples.loc[b, "y"] = 1 - data.samples.loc[b, "y"]
        # Validation-only values never change fitting filters or aggregate membership.
        after = construct(data, a, ids, Bacteria())[0]
        np.testing.assert_array_equal(after[a].toarray(), family[a].toarray())
        np.testing.assert_array_equal(Preprocessing().fit(data.values, a)[0], ids)
        np.testing.assert_array_equal(selected["order"], pipeline.selection(data, context, before, Bacteria(), 3)[1]["order"])
        self.assertTrue(np.all(np.asarray(data.values.sum(axis=1)).ravel() != 1))

    def test_invalid_data_constants_and_empty_predictors(self):
        data = self.dataset()
        bad = data.values.copy()
        bad.data[0] = -1
        with self.assertRaises(ValueError):
            validate(bad, data.samples, data.pair_bacteria, data.pair_gene)
        with self.assertRaises(ValueError):
            validate(data.values, data.samples, data.pair_bacteria.astype(float), data.pair_gene)
        with self.assertRaises(ValueError):
            load("../escape")
        with self.assertRaises(ValueError):
            retained_pair_values(sparse.csr_matrix([[0, 1], [0, 2]]), [0])
        np.testing.assert_array_equal(predict_model(sparse.csr_matrix((4, 0)), [0, 0, 0, 1], sparse.csr_matrix((2, 0))), [.25, .25])
        with self.assertRaises(ValueError):
            pipeline.contexts(np.array([0, 0, 0, 0, 1]), Evaluation(folds=2))
        for component in (lambda: BernoulliGraph(threshold=0), lambda: BernoulliGraph(memory_gib=-1),
                          lambda: Leiden(resolution=0), lambda: Leiden(iterations=0)):
            with self.assertRaises(ValueError):
                component()
        invalid = Experiment("bad", ("toy",), (Bacteria(),), ("discoveries",), ("negative_log10_p_overview.png",))
        with patch.object(pipeline, "load", side_effect=AssertionError("Loaded before validating outputs")):
            with self.assertRaises(ValueError):
                pipeline.run(invalid)

    def test_input_export_orientation_and_independent_participants(self):
        source = self.root / "source"
        source.mkdir()
        (source / "study_genes.txt").write_text("g0|b0\ng1|b1\ng2|unclassified\n")
        (source / "study_samples.txt").write_text("a\nb\nc\nd\n")
        metadata = pd.DataFrame(dict(sample_id=list("abcd"), subject_id=["p0", "p0", "p1", "p2"],
            body_site=["stool"] * 4, days_from_first_collection=[0, 0, 0, 1],
            number_reads=[10, 20, 10, 10], target=["no", "no", "yes", "yes"]))
        metadata.to_csv(source / "study_metadata.tsv", sep="\t", index=False)
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]])
        mmwrite(source / "study_gene_families.mtx", sparse.coo_matrix(values))
        output = prepare("exported", source, "study", target_column="target", negative_label="no",
                         positive_label="yes", independent_day_zero=True)
        result = load("exported")
        np.testing.assert_array_equal(result.values.toarray(), values[:2, [1, 2]].T)
        self.assertEqual(result.samples.sample_id.tolist(), ["b", "c"])
        self.assertEqual(result.pair_bacteria.tolist(), [0, 1])

    def test_new_cohort_export_counts_values_and_resumption(self):
        source = self.root / 'sources/YachidaS_2019'
        source.mkdir(parents=True)
        (source / 'YachidaS_2019_genes.txt').write_text('g0|b0\ng1|b1\ng2|unclassified\n')
        (source / 'YachidaS_2019_samples.txt').write_text('a\nb\nc\nd\n')
        pd.DataFrame(dict(sample_id=list('abcd'), subject_id=['p0', 'p1', 'p2', 'p3'],
            disease=['healthy', 'healthy', 'CRC', 'CRC'])).to_csv(source / 'YachidaS_2019_metadata.tsv', sep='\t', index=False)
        values = np.array([[1., 2, 3, 4], [.1, .2, 0, .4], [9, 9, 9, 9]])
        mmwrite(source / 'YachidaS_2019_gene_families.mtx', sparse.coo_matrix(values))
        with patch.object(acquire, 'SOURCE', source.parent), patch.object(acquire, 'DOWNLOADS', self.root / 'downloads'), \
             patch.dict(acquire.COHORTS, {'YachidaS_2019': ('disease', 'healthy', 'CRC', 2, 2)}), \
             patch.object(acquire, 'conversion_versions', return_value={'fixture': True}):
            acquire.prepare_one('YachidaS_2019', pd.DataFrame(), pd.DataFrame())
            result = load('YachidaS_2019')
            np.testing.assert_array_equal(result.values.toarray(), values[:2].T)
            folder = cache.CACHE / 'inputs/YachidaS_2019'
            with patch('datasets.prepare.mmread', side_effect=AssertionError('Reimported completed cohort')):
                acquire.prepare_one('YachidaS_2019', pd.DataFrame(), pd.DataFrame())
            protocol = json.loads((folder / 'protocol.json').read_text())
            protocol['code']['datasets.acquire.read_sparse_rds'] = 'unused old converter'
            protocol['files'][str(Path(acquire.__file__).with_name('read_cmd.R'))] = 'unused old reader'
            old_key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            cache.write_json(folder / 'protocol.json', protocol)
            cache.seal(folder, old_key, [p for p in folder.iterdir() if p.name != 'complete.json'])
            with patch('datasets.prepare.mmread', side_effect=AssertionError('Reimported completed cohort')):
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    acquire.prepare_one('YachidaS_2019', pd.DataFrame(), pd.DataFrame())
            self.assertFalse((folder / 'compatibility.json').exists())
            damaged = dict(protocol, settings=dict(protocol['settings'], positive='changed target'))
            cache.write_json(folder / 'protocol.json', damaged)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                acquire.prepare_one('YachidaS_2019', pd.DataFrame(), pd.DataFrame())
            with self.assertRaisesRegex(ValueError, 'labels changed'):
                acquire.check_counts(result.samples.iloc[:-1], 'YachidaS_2019')

    def test_approved_metacardis_zero_classified_exclusions(self):
        name = 'MetaCardis_2020_a'
        folder = self.root / name
        folder.mkdir()
        ids = [f's{i}' for i in range(18)]
        (folder / f'{name}_samples.txt').write_text('\n'.join(ids) + '\n')
        (folder / f'{name}_genes.txt').write_text('g0|b0\ng1|unclassified\n')
        pd.DataFrame(dict(sample_id=ids, disease=['T2D'] * 12 + ['healthy'] * 4 + ['T2D', 'healthy'])).to_csv(
            folder / f'{name}_metadata.tsv', sep='\t', index=False)
        values = np.vstack([np.r_[np.zeros(16), [2.5, 7]], np.ones(18)])
        mmwrite(folder / f'{name}_gene_families.mtx', sparse.coo_matrix(values))
        matrix, _, samples, _, omitted = acquire.metacardis_source(folder)
        np.testing.assert_array_equal(matrix.toarray(), values[:, -2:])
        self.assertEqual(samples, ids[-2:])
        self.assertEqual(omitted.disease.value_counts().to_dict(), {'T2D': 12, 'healthy': 4})

    def test_base_r_sparse_slots_and_arrow_cohort_extraction(self):
        if not acquire.R.exists() or not acquire.PARQUET_PYTHON.exists():
            self.skipTest('Optional installed source converters unavailable')
        rds = self.root / 'fixture.rds'
        for kind, second, contents, container in [
            (kind, slot, contents, container)
            for kind, slot, contents in [('dgCMatrix', 'p', 'c(0L,2L,3L)'), ('dgTMatrix', 'j', 'c(0L,0L,1L)')]
            for container in ('rds', 'rdata')]:
            expression = ('args<-commandArgs(TRUE);setClass("' + kind + '",slots=c(i="integer",' + second +
                '="integer",Dim="integer",Dimnames="list",x="numeric"));'
                'x<-new("' + kind + '",i=c(0L,1L,1L),' + second + '=' + contents + ',Dim=c(2L,2L),'
                'Dimnames=list(c("g0|b0","g1|b1"),c("a","b")),x=c(2,.5,7));'
                + ('saveRDS(x,args[1])' if container == 'rds' else 'save(x,file=args[1],compress="bzip2")'))
            subprocess.run([str(acquire.R), '-e', expression, str(rds)], check=True)
            directory = self.root / (kind + container)
            matrix, features, samples = acquire.read_sparse_rds(rds, directory)
            np.testing.assert_array_equal(matrix.toarray(), [[2, 0], [.5, 7]])
            self.assertEqual(features, ['g0|b0', 'g1|b1'])
            self.assertEqual(samples, ['a', 'b'])
            acquire.read_sparse_rds(rds, directory)
        expression = '''import sys, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from scipy import sparse
from datasets.read_parquet import extract
p=Path(sys.argv[1]);pd.DataFrame(dict(sample_id=['a','b'],uuid=['u0','u1'])).to_csv(p/'meta.csv',index=False)
pq.write_table(pa.table(dict(uuid=['u0','other','u1','u0'],gene_family=['g0|b0','g9|b9','g1|b1','g9|unclassified'],rpk_abundance=[.01,99.,.02,88.])),p/'fixture.parquet',row_group_size=1)
extract(p/'fixture.parquet',p/'meta.csv',p)
assert sparse.load_npz(p/'values.npz').toarray().tolist()==[[.01,0],[0,.02]]
'''
        subprocess.run([str(acquire.PARQUET_PYTHON), '-c', expression, str(self.root)],
                       cwd=Path(__file__).resolve().parents[1], check=True)

    def test_mw_by_and_masked_correlation_family(self):
        a = np.array([-1, -1, 1, 1])
        c = np.array([-1, 1, -1, 1])
        matrix = sparse.csr_matrix(np.column_stack((a + 2, a + 3, c + 2, a + c + 5, a - c + 5)))
        p = np.array([.001, .004, .020, .4, .9])
        np.testing.assert_allclose(BY().adjust(p)[:3], [.0114166666666667, .0228333333333333, .0761111111111111])
        selector = MannWhitney(correlation_threshold=.8)
        selected = selector.select({"p": p}, matrix, np.arange(5), 5, 75)
        np.testing.assert_array_equal(selected["native"], [0])
        np.testing.assert_array_equal(selected["order"], [0, 2, 3, 4])
        accepted = greedy_filter(matrix, np.arange(5), .8)
        working = np.ones(5)
        working[accepted] = p[accepted]
        np.testing.assert_array_equal(np.flatnonzero(BY().adjust(working) <= .1), selected["native"])
        with self.assertRaises(ValueError):
            greedy_filter(sparse.csr_matrix(np.ones((4, 1))), [0], .8)
        self.assertEqual(len(greedy_filter(matrix, [], .8)), 0)
        values = np.array([[0, 1], [1, 0], [2, 2], [3, 3.]])
        actual, _ = mann_whitney_many(sparse.csr_matrix(values), [np.array([0, 0, 1, 1])])
        for j in range(2):
            self.assertEqual(actual[0][j], stats.mannwhitneyu(values[:2, j], values[2:, j], alternative="two-sided", method="asymptotic").pvalue)

    def test_prepared_mw_matches_scipy_with_ties_unequal_groups_and_threads(self):
        values = sparse.csr_matrix(np.array([
            [0, 1, 0, 0], [0, 1, 1, 0], [1, 2, 1, 1],
            [1, 3, 2, 1], [3, 3, 0, 3], [5, 4, 3, 5],
        ], dtype=float))
        targets = np.array([np.isin(np.arange(6), group).astype(int)
            for size in range(1, 6) for group in itertools.combinations(range(6), size)])
        expected = np.asarray(mann_whitney_many(values, targets)[0])
        for workers in (1, 24):
            prepared = prepare_mann_whitney(values, workers=workers)
            np.testing.assert_array_equal(prepared.test(targets), expected)
            np.testing.assert_array_equal(prepared.test(1 - targets), expected)
            self.assertEqual(prepared.ranks.nnz, values.nnz)
        self.assertEqual(prepare_mann_whitney(values[:, :0]).test(targets).shape, (62, 0))
        for invalid in (sparse.csr_matrix(np.zeros((6, 1))), sparse.csr_matrix(np.ones((6, 1))),
                        values.astype(np.float32), -values):
            with self.assertRaises(ValueError):
                prepare_mann_whitney(invalid)
        for invalid in (np.ones((1, 6)), np.full((1, 6), np.nan), np.ones((1, 5))):
            with self.assertRaises(ValueError):
                prepared.test(invalid)

    def test_prepared_mw_checkpoint_prefix_and_implementation_rejection(self):
        data = self.dataset()
        family = pipeline.Family('fixture', self.root, data.values[:, 1:3].astype(float), np.array([1, 2]), 4, {})
        method = Gene()
        evaluation = Evaluation(folds=2, fine_k=(1,), coarse_k=(1,), shuffles=5)
        expected, keys = pipeline.shuffled_counts(data, family, [method], evaluation)
        directory = cache.CACHE / 'stages/toy/full/shuffles' / keys[method.name]
        (directory / 'complete.json').unlink()
        (directory / 'counts.npy').unlink()
        partial = expected[method.name].copy()
        partial[2:] = -1
        np.save(directory / 'checkpoint/counts.npy', partial)
        cache.seal(directory / 'checkpoint', keys[method.name], [directory / 'checkpoint/counts.npy'])
        actual, _ = pipeline.shuffled_counts(data, family, [method], evaluation)
        np.testing.assert_array_equal(actual[method.name], expected[method.name])
        before = (directory / 'counts.npy').stat().st_mtime_ns
        with patch('methods.selectors.mann_whitney.ThreadPoolExecutor', side_effect=AssertionError('Recomputed completed shuffles')):
            pipeline.shuffled_counts(data, family, [method], evaluation)
        self.assertEqual((directory / 'counts.npy').stat().st_mtime_ns, before)
        protocol = directory / 'protocol.json'
        changed = json.loads(protocol.read_text())
        changed['settings']['budget'] = 999
        cache.write_json(protocol, changed)
        with self.assertRaisesRegex(ValueError, 'Changed stage protocol'):
            pipeline.shuffled_counts(data, family, [method], evaluation)

    def test_exact_bins_full_catalog_and_native_ties(self):
        x = sparse.csr_matrix([[0, 1], [0, 1], [1, 2], [2, 3], [4, 4], [8, 5.]])
        targets = np.array([np.isin(np.arange(6), s).astype(int) for s in itertools.combinations(range(6), 3)])
        log_e = bins.evidence(bins.prepare_bins(x), targets)
        np.testing.assert_allclose(np.exp(log_e).mean(axis=0), 1, atol=1e-12)
        np.testing.assert_array_equal(EBH().select(np.log([30.]), [0], 1), [0])
        self.assertEqual(len(EBH().select(np.log([30.]), [0], 1000)), 0)
        np.testing.assert_array_equal(EBH().select(np.log([30., 30.]), [1, 0], 2), [1, 0])
        for ids, size in (([2], 2), ([-1], 2), ([.5], 2), ([0], .5)):
            with self.assertRaises(ValueError):
                EBH().select([1.], ids, size)
        self.assertEqual(bins.evidence(bins.prepare_bins(x[:, :0]), targets).shape, (20, 0))

    def test_stage_keys_reuse_graph_and_distinguish_cluster_components(self):
        data = self.dataset()
        context = pipeline.Context("full", np.arange(24), np.array([], int))
        original = BernoulliJoin(graph=TinyGraph(), clustering=TinyLeiden())
        first = pipeline.prepare_family(data, context, Preprocessing(), original)
        (self.root / "unused_selector.py").write_text("# not used")
        same = pipeline.prepare_family(data, context, Preprocessing(), original)
        self.assertEqual(first.key, same.key)
        self.assertEqual(TinyGraph.calls, 1)
        second = pipeline.prepare_family(data, context, Preprocessing(), replace(original, clustering=TinyLeiden(resolution=.5)))
        self.assertEqual(first.references["graph"], second.references["graph"])
        self.assertNotEqual(first.references["clustering"], second.references["clustering"])
        third = pipeline.prepare_family(data, context, Preprocessing(), replace(original, clustering=OtherLeiden()))
        self.assertNotEqual(first.references["clustering"], third.references["clustering"])
        self.assertEqual(TinyGraph.calls, 1)
        self.assertEqual(TinyLeiden.calls, 3)
        smaller_cap = pipeline.prepare_family(data, context, Preprocessing(),
            replace(original, graph=replace(original.graph, edge_limit=100, step=.01)))
        self.assertEqual(first.references['graph'], smaller_cap.references['graph'])
        self.assertNotEqual(first.references['clustering'], smaller_cap.references['clustering'])
        self.assertEqual(TinyGraph.calls, 1)

    def test_shuffles_alpha_guard_shared_batches_and_immutable_resumption(self):
        data = self.dataset()
        family = pipeline.Family("fixture", self.root, data.values[:, 1:2], np.array([1]), 4, {})
        baseline = BernoulliJoin(selector=FixedP(correction=BY(.01)))
        filtered = BernoulliJoin(selector=FixedP(correction=BY(.2), correlation_threshold=.8), name="filtered")
        evaluation = Evaluation(folds=2, fine_k=(1,), coarse_k=(1,), shuffles=5)
        result, keys = pipeline.shuffled_counts(data, family, [baseline, filtered], evaluation)
        np.testing.assert_array_equal(result[baseline.name], np.zeros(5))
        np.testing.assert_array_equal(result[filtered.name], np.ones(5))
        calls = FixedP.calls
        before = {p: p.stat().st_mtime_ns for p in cache.CACHE.rglob("counts.npy")}
        same, _ = pipeline.shuffled_counts(data, family, [baseline, filtered], evaluation)
        self.assertEqual(FixedP.calls, calls)
        zero_filter = replace(filtered, selector=FixedP(correction=BY(.01), correlation_threshold=.7))
        result, _ = pipeline.shuffled_counts(data, family, [baseline, zero_filter], evaluation)
        self.assertEqual(FixedP.calls, calls)
        self.assertEqual({p: p.stat().st_mtime_ns for p in before}, before)
        np.testing.assert_array_equal(result[zero_filter.name], np.zeros(5))
        targets = label_permutations(data.samples.y.to_numpy(), "toy/full_data", 5)
        np.testing.assert_array_equal(targets, label_permutations(data.samples.y.to_numpy(), "toy/full_data", 5))
        self.assertTrue(np.all(targets.sum(axis=1) == data.samples.y.sum()))

    def test_count_metrics_and_correlation_boundaries(self):
        self.assertEqual(discovery_statistics([0] * 999, 1, 100)["p_value"], .001)
        self.assertEqual(discovery_statistics([0] * 999, 0, 100)["p_value"], 1)
        self.assertEqual(discovery_statistics([1, 1, 0], 1, 10)["p_value"], .75)
        result = discovery_statistics([0, 1, 2], 10, 100)
        mean = (10 + 3) / 4
        z = (10 - 1) / np.sqrt(mean * (1 - mean / 100) * (1 + 1 / 3))
        self.assertAlmostEqual(result["normal_log_p_value"], stats.norm.logsf(z))
        x = sparse.csr_matrix([[0, 2], [1, 1], [2, 0.]])
        self.assertAlmostEqual(average_pairwise_absolute_correlation(x, [0, 1])["average_pairwise_absolute_correlation"], 1)
        self.assertEqual(average_pairwise_absolute_correlation(None, [])["status"], "undefined_subset_size")
        self.assertEqual(average_pairwise_absolute_correlation(None, range(10001))["status"], "skipped_size_limit")

    def test_graph_disk_reserve_is_explicit_and_matches_experiment_definitions(self):
        from experiments.main_comparison import EXPERIMENT
        from experiments.join_correlation_filter import experiment
        self.assertEqual(BernoulliGraph().free_disk_gib, 50)
        for definition in (EXPERIMENT, experiment([.8])):
            for method in definition.methods:
                if method.graph is not None:
                    self.assertEqual(method.graph.free_disk_gib, 50)

    def test_input_export_rejects_ambiguous_axes_and_missing_participants_before_writing(self):
        metadata = pd.DataFrame(dict(sample_id=['a', 'b', 'c'], subject_id=['p0', None, 'p2'],
                                     target=['no', 'yes', 'no']))
        arguments = dict(source_study='fixture', target_column='target', negative_label='no', positive_label='yes')
        destination = self.root / 'invalid_export'
        with self.assertRaisesRegex(ValueError, 'orientation ambiguous'):
            prepare_matrix('square', sparse.csr_matrix(np.ones((3, 3))), ['g0|b0', 'g1|b1', 'g2|b2'],
                           ['a', 'b', 'c'], metadata, destination=destination, **arguments)
        with self.assertRaisesRegex(ValueError, 'independent participants'):
            prepare_matrix('missing', sparse.csr_matrix(np.ones((2, 3))), ['g0|b0', 'g1|b1'],
                           ['a', 'b', 'c'], metadata, destination=destination, **arguments)
        self.assertFalse(destination.exists())

    def test_graph_guards_and_float32_boundary(self):
        self.assertEqual(partition.fallback_thresholds(.7), [.7, .75, .8, .85, .9, .95, 1.])
        self.assertEqual(partition.choose_effective_threshold({.7: 501, .75: 500}, 500), .75)
        shard = np.array([(np.float32(.8),), (np.float32(.79),)], dtype=[("phi", "f4")])
        counts = graph_preparation.count_threshold_edges(None, [.8], {"edge_weight": {"field": "phi", "transform": "abs"}}, lambda _: [shard])
        self.assertEqual(counts[.8], 1)
        with self.assertRaises(RuntimeError):
            partition.choose_effective_threshold({1.: 501}, 500)
        np.testing.assert_array_equal(partition.stable_cluster_labels([0, 1], [0, -1], [2, 4], 4), [0, 1])
        self.assertEqual(partition.fallback_thresholds(.735, .07), [.735, .77, .84, .91, .98, 1.])
        self.assertEqual(partition.fallback_thresholds(.9985, .001), [.9985, .999, 1.])
        ids, labels = np.array([8, 2, 9, 1]), np.array([20, 4, 20, 7])
        np.testing.assert_array_equal(partition.stable_cluster_labels(ids, labels, np.ones(4), 8), [2, 1, 2, 0])

    def test_partition_retries_only_memory_failures_on_the_same_source(self):
        from methods.clustering import community
        args = SimpleNamespace(output=self.root / 'partition', threshold=.7, threshold_step=.05,
            fixed_threshold=False, community_working_memory_gb=1, community_max_edges=1000,
            community_workers=1, seed=17729, resolution=1, memory_gb=1, output_gb=1,
            free_disk_gb=0, iterations=-1)
        source, filtered = self.root / 'source', self.root / 'higher_threshold'
        manifest = dict(threshold=.7, edge_count=1)
        fitted = (np.array([0]), np.array([0]), {'status': 'complete'})
        with patch.object(community, 'run_community_detection', side_effect=[MemoryError('fixture'), fitted]) as fit, \
             patch.object(partition, 'count_threshold_edges', return_value={.7: 1, .75: 1}), \
             patch.object(partition, 'threshold_graph', return_value=filtered) as materialize:
            _, _, result = partition.fit_partition(args, source, manifest)
        self.assertEqual([call.args[0] for call in fit.call_args_list], [source, filtered])
        self.assertEqual(materialize.call_args.args, (args, source, manifest, .75, 1))
        self.assertEqual(result['effective_threshold'], .75)
        self.assertEqual(len(result['resource_retries']), 1)
        with patch.object(community, 'run_community_detection', side_effect=RuntimeError('bad graph')), \
             patch.object(partition, 'threshold_graph') as materialize:
            with self.assertRaisesRegex(RuntimeError, 'bad graph'):
                partition.fit_partition(args, source, manifest)
            materialize.assert_not_called()
        args.fixed_threshold = True
        with patch.object(community, 'run_community_detection', side_effect=MemoryError('fixture')):
            with self.assertRaisesRegex(RuntimeError, 'Fixed-threshold Leiden'):
                partition.fit_partition(args, source, manifest)

    def test_invalid_new_shuffle_counts_cannot_be_published_complete(self):
        @dataclass(frozen=True)
        class InvalidSelector(FixedP):
            def select(self, *args):
                return {'native': np.array([0, 0])}
        data = self.dataset()
        family = pipeline.Family('fixture', self.root, data.values[:, 1:2], np.array([1]), 4, {})
        method = BernoulliJoin(selector=InvalidSelector())
        evaluation = Evaluation(folds=2, fine_k=(1,), coarse_k=(1,), shuffles=2)
        with self.assertRaisesRegex(ValueError, 'invalid shuffled counts'):
            pipeline.shuffled_counts(data, family, [method], evaluation)
        stages = list((cache.CACHE / 'stages/toy/full/shuffles').iterdir())
        self.assertTrue(stages)
        self.assertTrue(all(not (stage / 'complete.json').exists() for stage in stages))

    def test_local_exact_worker_signed_edges_and_leiden(self):
        from methods.graphs.exact import iter_edge_shards, run_join
        values = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 0], [1, 1, 0, 1, 0],
                           [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0]], float)
        ids = np.array([40, 10, 30, 20, 50])
        matrix, selection = self.root / "worker_matrix.npz", self.root / "worker_selection.npz"
        sparse.save_npz(matrix, sparse.csr_matrix(values))
        np.savez(selection, train_index=np.arange(8), feature_ids=np.arange(5))
        expected = {(0, 1): 1., (0, 2): -1., (1, 2): -1.}
        for workers in (1, 4):
            graph = BernoulliGraph(workers=workers, memory_gib=1, working_memory_gib=.1,
                                  reserve_gib=0, free_disk_gib=0, output_gib=.1)
            directory = self.root / f"actual_graph_{workers}"
            self.assertEqual(graph.command(matrix, selection, directory)[:4],
                             [sys.executable, "-u", "-m", "methods.graphs.worker"])
            graph.build(matrix, selection, directory)
            edges = {(int(e['feature_j']), int(e['feature_k'])): float(e['phi'])
                     for shard in iter_edge_shards(directory / 'graph') for e in shard}
            self.assertEqual(edges, expected)
            mtimes = {p: p.stat().st_mtime_ns for p in directory.rglob('*') if p.is_file()}
            output = self.root / f"actual_partition_{workers}"
            Leiden(working_memory_gib=1).fit(directory, matrix, selection, output, graph)
            with np.load(output / 'assignments.npz') as saved:
                np.testing.assert_array_equal(saved['clusters'], [0, 0, 0, 1, 2])
            self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
        packed = self.root / 'packed_failure'
        graph_preparation.pack_sparse_detection(sparse.csr_matrix(values), ids, packed)
        failed = run_join(packed, self.root / 'aborted', .7, memory_limit_gib=1,
                          output_limit_gib=1e-8, free_disk_reserve_gib=0, workers=1, quiet=True)
        self.assertEqual(failed['status'], 'aborted_resource_limit')
        numpy_settings = run_join(packed, self.root / 'numpy_settings', np.float32(.7),
            memory_limit_gib=np.float64(1), working_memory_gib=np.float32(.1), output_limit_gib=np.float32(.1),
            free_disk_reserve_gib=np.float32(0), workers=np.int64(1), min_one_support=np.int64(1), quiet=True)
        self.assertEqual(numpy_settings['status'], 'complete')
        json.dumps(numpy_settings)  # Metadata remains serializable without a global NumPy encoder.

    def test_partition_seeds_do_not_depend_on_shard_order(self):
        from methods.graphs.exact import NODE_DTYPE, EDGE_DTYPE
        from methods.clustering import community
        memberships, seeds = [], []
        for reverse in (False, True):
            graph = self.root / f'order_{reverse}'
            (graph / 'edges').mkdir(parents=True)
            nodes = np.zeros(4, dtype=NODE_DTYPE)
            nodes['feature_id'], nodes['eligible'] = [90, 5, 60, 2], True
            np.save(graph / 'nodes.npy', nodes)
            pairs = [(5, 90, 1.), (2, 60, 1.)]
            for i, edge in enumerate(pairs[::-1] if reverse else pairs):
                np.save(graph / 'edges' / f'part-{i:06d}.npy', np.array([edge], dtype=EDGE_DTYPE))
            cache.write_json(graph / 'run.json', dict(status='complete', edge_count=2,
                edge_weight=dict(field='phi', transform='abs')))
            seen = {}
            original = community._leiden_component
            def record(task):
                seen[int(nodes['feature_id'][np.unique(np.r_[task[1], task[2]])].min())] = task[0]
                return original(task)
            with patch.object(community, '_leiden_component', side_effect=record):
                row_ids, membership, _ = community.run_community_detection(graph, output_dir=graph / 'partition',
                    memory_limit_gib=1, working_memory_gib=1, free_disk_reserve_gib=0, quiet=True)
            np.testing.assert_array_equal(row_ids, nodes['feature_id'])
            memberships.append(membership)
            seeds.append(seen)
        np.testing.assert_array_equal(*memberships)
        self.assertEqual(seeds, [{2: 0, 5: 1}] * 2)

    def test_concurrent_shard_flushes_share_one_disk_reserve(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import time
        from methods.graphs import exact
        from methods.graphs.resources import ResourceLimit
        guard, start = threading.Lock(), threading.Barrier(4)
        class NearReserve:
            output_limit_bytes = 1024**3
            free = 400
            def check_memory(self, *args):
                pass
            def check_output(self, extra, label):
                with guard:
                    if extra > self.free:
                        raise ResourceLimit('near reserve')
                time.sleep(.01)  # Exposes a check/write race if the writer lock is missing.
        monitor = NearReserve()
        output = self.root / 'concurrent_writer'
        output.mkdir()
        writer = exact.EdgeWriter(output, monitor, np.zeros(2, dtype=exact.NODE_DTYPE))
        real_save = np.save
        def save(handle, values, **kwargs):
            real_save(handle, values, **kwargs)
            with guard:
                monitor.free -= handle.tell()
        def flush(_):
            start.wait()
            try:
                writer._flush_state(dict(buffer=np.zeros(3, dtype=exact.EDGE_DTYPE), used=3))
                return True
            except ResourceLimit:
                return False
        with patch.object(exact.np, 'save', side_effect=save), ThreadPoolExecutor(4) as pool:
            written = list(pool.map(flush, range(4)))
        self.assertEqual(sum(written), 1)
        self.assertEqual(writer.edge_count, 3)
        self.assertGreaterEqual(monitor.free, 0)

    def test_empty_constant_graph_keeps_singletons_and_guards_assignment_output(self):
        from methods.graphs.resources import ResourceLimit
        matrix, selection = self.root / 'constant.npz', self.root / 'constant_selection.npz'
        sparse.save_npz(matrix, sparse.csr_matrix(np.ones((8, 3))))
        np.savez(selection, train_index=np.arange(8), feature_ids=np.arange(3))
        graph = BernoulliGraph(workers=1, memory_gib=1, working_memory_gib=.1,
                              reserve_gib=0, free_disk_gib=0, output_gib=.1)
        source, output = self.root / 'constant_graph', self.root / 'constant_partition'
        graph.build(matrix, selection, source)
        Leiden(working_memory_gib=1).fit(source, matrix, selection, output, graph)
        with np.load(output / 'assignments.npz') as saved:
            np.testing.assert_array_equal(saved['clusters'], [0, 1, 2])
            np.testing.assert_array_equal(saved['cluster_sizes'], [1, 1, 1])
        self.assertFalse(list(output.rglob('*.csv*')))
        self.assertFalse((source / 'graph/timings.jsonl').exists())
        args = SimpleNamespace(output=self.root / 'too_small', graph_source=source,
                               memory_gb=1, output_gb=1e-8, free_disk_gb=0)
        fitted = np.arange(3), np.full(3, -1), {}
        with patch.object(partition, 'fit_partition', return_value=fitted):
            with self.assertRaises(ResourceLimit):
                partition.cluster_graph(args, np.arange(8), np.arange(3))
        self.assertFalse((args.output / 'assignments.npz').exists())
        with self.assertRaisesRegex(RuntimeError, 'original feature IDs'):
            partition.write_assignments(output, np.arange(3), np.full(3, 8), 8,
                                        np.array([0, 1, 5]), np.arange(3))

    def test_bounded_batches_and_small_shards_match_dense_signed_oracle(self):
        from methods.graphs import exact
        rng = np.random.default_rng(33)
        base = (rng.random(129) < .5).astype(np.int8)
        values = np.tile(base[:, None], (1, 40))
        values[:, 10:20] = 1 - values[:, 10:20]
        for j in range(20, 40):
            values[rng.choice(129, 3, replace=False), j] ^= 1
        ids = rng.permutation(40)
        packed = self.root / 'bounded_packed'
        graph_preparation.pack_sparse_detection(sparse.csr_matrix(values), ids, packed)
        correlations = np.corrcoef(values, rowvar=False)
        expected = {tuple(sorted((int(ids[j]), int(ids[k])))): correlations[j, k]
                    for j in range(40) for k in range(j + 1, 40) if abs(correlations[j, k]) >= .7}
        for workers in (1, 4):
            graph = self.root / f'bounded_{workers}'
            with patch.object(exact, 'DEFAULT_BUFFER_BYTES', 84), patch.object(exact, 'GROUP_BATCH_SIZE', 11):
                result = exact.run_join(packed, graph, .7, workers=workers, memory_limit_gib=1,
                    working_memory_gib=.1, output_limit_gib=.1, free_disk_reserve_gib=0, quiet=True)
            self.assertEqual(result['status'], 'complete')
            self.assertGreater(result['edge_shards'], 1)
            actual = {(int(e['feature_j']), int(e['feature_k'])): float(e['phi'])
                      for shard in exact.iter_edge_shards(graph) for e in shard}
            self.assertEqual(set(actual), set(expected))
            for pair in expected:
                self.assertAlmostEqual(actual[pair], expected[pair], places=6)

    def test_exact_signed_phi_matches_brute_force_across_words_and_workers(self):
        from methods.graphs.exact import iter_edge_shards, run_join
        rng = np.random.default_rng(42)
        values = (rng.random((70, 24)) < rng.uniform(.1, .9, 24)).astype(np.int8)
        values[:, 1], values[:, 2] = values[:, 0], 1 - values[:, 0]
        values[:, 3] = values[:, 0]
        values[[2, 11, 35], 3] ^= 1
        values[:, -1] = 1
        ids = rng.permutation(24) + 100
        packed = self.root / 'packed_oracle'
        graph_preparation.pack_sparse_detection(sparse.csr_matrix(values), ids, packed)
        correlations = np.corrcoef(values[:, :-1], rowvar=False)
        self.assertTrue(np.any((np.abs(correlations) > .85) & (np.abs(correlations) < .999)))
        for threshold, workers in itertools.product((.6, .85), (1, 4)):
            graph = self.root / f'oracle_{threshold}_{workers}'
            result = run_join(packed, graph, threshold, workers=workers, memory_limit_gib=1,
                              working_memory_gib=.1, output_limit_gib=.1, free_disk_reserve_gib=0, quiet=True)
            self.assertEqual(result['status'], 'complete')
            actual = {(int(e['feature_j']), int(e['feature_k'])): float(e['phi'])
                      for shard in iter_edge_shards(graph) for e in shard}
            expected = {tuple(sorted((int(ids[j]), int(ids[k])))): correlations[j, k]
                        for j in range(23) for k in range(j + 1, 23) if abs(correlations[j, k]) >= threshold}
            self.assertEqual(set(actual), set(expected))
            for pair in actual:
                self.assertAlmostEqual(actual[pair], expected[pair], places=6)

    def test_shared_pipeline_lock_rejects_another_writer_and_releases_on_failure(self):
        cache.CACHE.mkdir(parents=True)
        with (cache.CACHE / '.pipeline.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another scientific experiment'):
                pipeline.run(None)
        with patch.object(pipeline, '_run', side_effect=ValueError('fixture failure')):
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                pipeline.run(None)
        with patch.object(pipeline, '_run', return_value=[]):
            self.assertEqual(pipeline.run(None), [])

    def test_bins_calibration_is_reused_across_shuffle_batches(self):
        x = sparse.csr_matrix([[0, 1], [0, 1], [1, 2], [2, 3], [4, 4], [8, 5.]])
        target = np.array([[0, 0, 0, 1, 1, 1]])
        prepared = bins.prepare_bins(x)
        expected = bins.evidence(prepared, target)
        with patch.object(bins, 'normalizer', side_effect=AssertionError('Recomputed invariant normalizers')):
            np.testing.assert_array_equal(bins.evidence(prepared, target), expected)
        with self.assertRaisesRegex(ValueError, 'unchanged label-group'):
            bins.evidence(prepared, np.array([[0, 0, 1, 1, 1, 1]]))

    def test_graph_resumes_only_explicit_unsealed_resource_abort(self):
        @dataclass(frozen=True)
        class AbortingGraph(TinyGraph):
            status: ClassVar[str] = 'aborted_resource_limit'

            def build(self, matrix, selection, output):
                if type(self).status == 'complete':
                    return super().build(matrix, selection, output)
                cache.write_json(output / 'graph/run.json', {'status': type(self).status})
                raise subprocess.CalledProcessError(1, ['fixture graph'])

        data = self.dataset()
        context = pipeline.Context('full', np.arange(len(data.samples)), np.array([], int))
        method = BernoulliJoin(graph=AbortingGraph(), clustering=TinyLeiden())
        with self.assertRaises(subprocess.CalledProcessError):
            pipeline.prepare_family(data, context, Preprocessing(), method)
        graph = next((cache.CACHE / 'stages/toy/full/graph').iterdir())
        work = graph / 'work'
        manifest = work / 'graph/run.json'
        aborted = manifest.read_bytes()
        AbortingGraph.status = 'complete'
        for status in ('complete', 'failed'):
            cache.write_json(manifest, {'status': status})
            with self.assertRaises(FileExistsError):
                pipeline.prepare_family(data, context, Preprocessing(), method)
            self.assertEqual(json.loads(manifest.read_text())['status'], status)
        manifest.unlink()
        with self.assertRaises(FileExistsError):
            pipeline.prepare_family(data, context, Preprocessing(), method)
        manifest.write_text('{malformed')
        with self.assertRaises(json.JSONDecodeError):
            pipeline.prepare_family(data, context, Preprocessing(), method)
        manifest.write_bytes(aborted)
        for index, path in enumerate((work, work / 'graph', manifest)):
            protected = self.root / f'protected_{index}'
            path.rename(protected)
            path.symlink_to(protected, target_is_directory=protected.is_dir())
            with self.assertRaisesRegex(ValueError, 'outside its stage'):
                pipeline.prepare_family(data, context, Preprocessing(), method)
            self.assertTrue(protected.exists())
            path.unlink()
            protected.rename(path)
        builds = TinyGraph.calls
        family = pipeline.prepare_family(data, context, Preprocessing(), method)
        self.assertEqual(json.loads((graph / 'aborted_run.json').read_text()), json.loads(aborted))
        self.assertFalse((graph / 'work/graph/run.json').exists())
        # A completed graph is still reused and never moved aside.
        pipeline.prepare_family(data, context, Preprocessing(), method)
        self.assertEqual(TinyGraph.calls, builds + 1)
        self.assertEqual(family.references['graph'], graph.name)
        (graph / 'work/graph.json').write_text('damaged')
        with self.assertRaisesRegex(ValueError, 'Changed stage artifact'):
            pipeline.prepare_family(data, context, Preprocessing(), method)

    def test_metric_protocol_rejects_changed_metrics_but_allows_plot_styling(self):
        experiment = Experiment("metric_fixture", ("toy",), (Bacteria(),), ("correlation",),
                                ("average_pairwise_absolute_correlation_overview.png",))
        output = self.root / "outputs/metric_fixture/.cache"
        from dataclasses import asdict
        cache.write_json(output / "experiment.json", asdict(experiment))
        cache.write_json(output / "metric_protocol.json", cache.metric_protocol(experiment)[1])
        pd.DataFrame([dict(dataset="toy", method="Bacteria-collapsed",
            average_pairwise_absolute_correlation=.25, status="defined")]).to_csv(output / "full_metrics.csv", index=False)
        palette = comparison.plt.get_cmap("Set2")
        with patch.object(comparison.plt, "get_cmap", return_value=palette):
            comparison.render(experiment)
        with patch.object(cache, "file_hash", return_value="changed metric implementation"):
            with self.assertRaisesRegex(ValueError, "metric definitions changed"):
                comparison.render(experiment)

    def test_failed_rerun_preserves_saved_tables_protocol_and_figures(self):
        data = self.dataset()
        experiment = Experiment('preserved', ('toy', 'missing'), (Bacteria(),), ('discoveries',),
                                ('real_discoveries_overview.png',))
        output = self.root / 'outputs/preserved'
        tables = output / '.cache'
        cache.write_json(tables / 'experiment.json', asdict(experiment))
        cache.write_json(tables / 'metric_protocol.json', cache.metric_protocol(experiment)[1])
        pd.DataFrame([dict(dataset='toy', method='Bacteria-collapsed', real_discoveries=7)]).to_csv(
            tables / 'full_metrics.csv', index=False)
        (tables / 'artifacts.csv').write_text('prior validated stage references\n')
        (output / experiment.outputs[0]).write_bytes(b'prior figure')
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tables.iterdir()}
        figure = output / experiment.outputs[0]
        before[figure] = (figure.read_bytes(), figure.stat().st_mtime_ns)
        with patch.object(pipeline, 'load', side_effect=[data, ValueError('Incompatible stage')]), \
             patch.object(comparison, 'render', side_effect=AssertionError('Rendered partial results')):
            failures = pipeline.run(experiment)
        self.assertEqual(failures, [dict(dataset='missing', error='ValueError: Incompatible stage')])
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}, before)
        self.assertFalse(list(output.glob('.tables_*')))
        self.assertEqual(json.loads((tables / 'status.json').read_text())['state'], 'incomplete')

    def test_render_failure_preserves_saved_results(self):
        self.dataset()
        experiment = Experiment('render_failure', ('toy',), (Bacteria(),), ('discoveries',),
                                ('real_discoveries_overview.png',))
        output = self.root / 'outputs/render_failure'
        tables = output / '.cache'
        cache.write_json(tables / 'experiment.json', asdict(experiment))
        cache.write_json(tables / 'metric_protocol.json', cache.metric_protocol(experiment)[1])
        pd.DataFrame([dict(dataset='toy', method='Bacteria-collapsed', real_discoveries=7)]).to_csv(
            tables / 'full_metrics.csv', index=False)
        (output / experiment.outputs[0]).write_bytes(b'prior figure')
        paths = list(tables.iterdir()) + [output / experiment.outputs[0]]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

        def fail_render(definition, *, output):
            (output / definition.outputs[0]).write_bytes(b'partial replacement')
            raise RuntimeError('Figure save failed')

        with patch.object(comparison, 'render', side_effect=fail_render):
            with self.assertRaisesRegex(RuntimeError, 'Figure save failed'):
                pipeline.run(experiment)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)
        status = json.loads((tables / 'status.json').read_text())
        self.assertEqual(status['stage'], 'reporting')
        self.assertTrue(status['retained_previous_results'])
        self.assertFalse(list(output.glob('.tables_*')))

    def test_table_write_failure_records_status_and_preserves_results(self):
        experiment = Experiment('disk_failure', ('toy',), (Bacteria(),), ('discoveries',),
                                ('real_discoveries_overview.png',))
        output = self.root / 'outputs/disk_failure'
        tables = output / '.cache'
        cache.write_json(tables / 'experiment.json', asdict(experiment))
        cache.write_json(tables / 'metric_protocol.json', cache.metric_protocol(experiment)[1])
        (tables / 'full_metrics.csv').write_text('prior validated table\n')
        (output / experiment.outputs[0]).write_bytes(b'prior figure')
        paths = list(tables.iterdir()) + [output / experiment.outputs[0]]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        with patch.object(pipeline, 'run_tables', side_effect=OSError('disk full')), \
             patch.object(comparison, 'render') as render:
            with self.assertRaisesRegex(OSError, 'disk full'):
                pipeline.run(experiment)
            render.assert_not_called()
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)
        status = json.loads((tables / 'status.json').read_text())
        self.assertEqual((status['state'], status['stage'], status['error']), ('failed', 'tables', 'OSError: disk full'))
        self.assertTrue(status['retained_previous_results'])
        self.assertFalse(list(output.glob('.tables_*')))

    def test_small_end_to_end_four_methods_cache_resumption_and_plot_only(self):
        self.dataset()
        evaluation = Evaluation(folds=2, fine_k=(1, 2), coarse_k=(2,), shuffles=3)
        methods = (Bacteria(), Gene(), BernoulliJoin(graph=TinyGraph(), clustering=TinyLeiden()), BinsEValue())
        experiment = Experiment("fixture", ("toy",), methods, ("prediction", "discoveries", "correlation", "shuffled_fdp"),
                                ("AUC_overview_k_1-2.png", "real_discoveries_overview.png"), evaluation=evaluation)
        self.assertEqual(pipeline.run(experiment), [])
        output = self.root / "outputs/fixture"
        self.assertEqual({p.name for p in output.glob("*.png")}, set(experiment.outputs))
        folds = pd.read_csv(output / ".cache/fold_metrics.csv")
        predictions = pd.read_csv(output / ".cache/predictions.csv.gz")
        summary = pd.read_csv(output / ".cache/prediction_summary.csv")
        self.assertEqual(len(summary), 8)
        self.assertEqual(len(pd.read_csv(output / ".cache/full_metrics.csv")), 4)
        self.assertEqual(len(pd.read_csv(output / ".cache/artifacts.csv")), 12)
        partial = folds[~((folds.method == methods[0].name) & (folds.fold == 1))]
        self.assertNotIn(methods[0].name, set(summarize(partial, predictions, evaluation).method))
        builds = TinyGraph.calls
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Refit completed cache")):
            self.assertEqual(pipeline.run(experiment), [])
        self.assertEqual(TinyGraph.calls, builds)
        with patch.object(pipeline, "run", side_effect=AssertionError("Plot requested science")):
            comparison.render(experiment)
        self.assertEqual(len(list(output.glob("*.png"))), 2)
        stale = replace(experiment, methods=methods[:1])
        with self.assertRaises(ValueError):
            comparison.render(stale)


if __name__ == "__main__":
    unittest.main()
