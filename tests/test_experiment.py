"""Small supplied-value, inference, stage-cache, and reporting fixtures only."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch
import itertools
import hashlib
import json
import tempfile
import unittest
import subprocess
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.io import mmwrite
from datasets.loading import load, validate
from datasets.prepare import prepare
from datasets import acquire
from execution import cache, pipeline
from execution.spec import Experiment, Evaluation
from methods.preprocessing import Preprocessing, retained_pair_values
from methods.pipelines import Bacteria, Gene, BernoulliJoin, BinsEValue, construct
from methods.graphs.bernoulli_join import BernoulliGraph
from methods.clustering.leiden import Leiden
from methods.graphs import worker
from methods.selectors import bins
from methods.selectors.mann_whitney import MannWhitney, mann_whitney_many
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
                                    (comparison, "PROJECT", self.root), (pipeline, "JOIN_PROJECT", self.root / "external")):
            p = patch.object(module, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.external = self.root / "external"
        self.external.mkdir()
        for name in ("bernoulli_join.py", "community_detection.py"):
            (self.external / name).write_text("# fixture")
        p = patch.object(pipeline, "external_versions", return_value={"python": "fixture", "numpy": "fixture", "igraph": "fixture"})
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
            protocol = json.loads((folder / 'protocol.json').read_text())
            protocol['code']['datasets.acquire.read_sparse_rds'] = 'unused old converter'
            protocol['files'][str(Path(acquire.__file__).with_name('read_cmd.R'))] = 'unused old reader'
            old_key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            cache.write_json(folder / 'protocol.json', protocol)
            cache.seal(folder, old_key, [p for p in folder.iterdir() if p.name != 'complete.json'])
            with patch('datasets.prepare.mmread', side_effect=AssertionError('Reimported completed cohort')):
                acquire.prepare_one('YachidaS_2019', pd.DataFrame(), pd.DataFrame())
            self.assertEqual(json.loads((folder / 'compatibility.json').read_text())['original_signature'], old_key)
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

    def test_extension_disk_override_preserves_scientific_settings(self):
        from experiments import extend_datasets as extension
        graph = BernoulliGraph()
        before = dict(graph.__dict__)
        for mode in ('graph', 'cluster'):
            command = graph.command('matrix.npz', 'selection.npz', 'work') + ['--mode', mode]
            expected = command.copy()
            expected[expected.index('--free-disk-gb') + 1] = '100'
            with patch.object(extension.Path, 'read_text', return_value='MemAvailable: 200000000 kB'), \
                    patch.object(extension.shutil, 'disk_usage') as disk, \
                    patch.object(extension, 'ORIGINAL_RUN') as run, \
                    patch.object(extension.time, 'sleep', side_effect=AssertionError('Unnecessary wait')):
                disk.return_value.free = 150 * 1024**3
                extension.graph_run(command, check=True)
                run.assert_called_once_with(expected, check=True)
            self.assertEqual(command[command.index('--free-disk-gb') + 1], '500')
        self.assertEqual(graph.__dict__, before)

    def test_graph_guards_and_float32_boundary(self):
        self.assertEqual(worker.fallback_thresholds(.7), [.7, .75, .8, .85, .9, .95, 1.])
        self.assertEqual(worker.choose_effective_threshold({.7: 501, .75: 500}, 500), .75)
        shard = np.array([(np.float32(.8),), (np.float32(.79),)], dtype=[("phi", "f4")])
        counts = worker.count_threshold_edges(None, [.8], {"edge_weight": {"field": "phi", "transform": "abs"}}, lambda _: [shard])
        self.assertEqual(counts[.8], 1)
        with self.assertRaises(RuntimeError):
            worker.choose_effective_threshold({1.: 501}, 500)
        np.testing.assert_array_equal(worker.stable_cluster_labels([0, 1], [0, -1], [2, 4], 4), [0, 1])

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
