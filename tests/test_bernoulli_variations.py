"""Small invariants for the cached Bernoulli-join variation study."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.bernoulli_coproject import (
    canonical_labels, contract_graph, contracted_ids, initial_graph,
)
from scripts.run_bernoulli_variations import (
    JOIN_PYTHON, _clustered_marginals, aligned_labels, choose_representatives,
    graph_path, signed_features,
    split_by_bacterium, strongest_negative_edges,
)
from scripts.summarize_bernoulli_variations import summarize_rows


class VariationTests(unittest.TestCase):
    def _pair_graph(self, root):
        graph = root / "pair_graph"
        (graph / "edges").mkdir(parents=True)
        edges = np.array([
            (0, 1, .8), (2, 3, .7), (2, 4, -.9),
            (0, 2, .95), (1, 4, .85),
        ], dtype=[("feature_j", "<u4"), ("feature_k", "<u4"), ("phi", "<f4")])
        np.save(graph / "edges/part-000000.npy", edges)
        (graph / "run.json").write_text(json.dumps({
            "status": "complete", "threshold": .7, "edge_count": len(edges),
            "edge_weight": {"field": "phi", "transform": "abs"},
        }))
        return graph

    @staticmethod
    def _edges(graph):
        return np.concatenate([np.load(path) for path in sorted((graph / "edges").glob("*.npy"))])

    def test_initial_projection_uses_possible_pair_density_and_self_loops(self):
        pair_bacteria = np.array([0, 0, 1, 1, 2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._pair_graph(root)
            positive = initial_graph(
                source, root / "positive", pair_bacteria, np.arange(5),
                "positive", "bacteria", free_disk_reserve_gib=0)
            absolute = initial_graph(
                source, root / "absolute", pair_bacteria, np.arange(5),
                "absolute", "bacteria", free_disk_reserve_gib=0)
            pos = self._edges(root / "positive")
            observed = {(int(row["feature_j"]), int(row["feature_k"])):
                        (float(row["numerator"]), float(row["weight"])) for row in pos}
            self.assertEqual(positive["edge_count"], 4)
            self.assertEqual(absolute["edge_count"], 5)
            self.assertAlmostEqual(observed[(0, 0)][1], .8)
            self.assertAlmostEqual(observed[(0, 1)][1], .95 / 4)
            self.assertAlmostEqual(observed[(0, 2)][1], .85 / 2)
            self.assertEqual(json.loads((source / "run.json").read_text())["edge_count"], 5)
            np.testing.assert_allclose(
                np.load(source / "edges/part-000000.npy")["phi"],
                [.8, .7, -.9, .95, .85])

    def test_contraction_matches_direct_projection_and_never_grows(self):
        pair_bacteria = np.array([0, 0, 1, 1, 2])
        pair_gene = np.array([0, 0, 1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._pair_graph(root)
            initial = initial_graph(
                source, root / "initial", pair_bacteria, np.arange(5),
                "absolute", "bacteria", free_disk_reserve_gib=0)
            contracted = contract_graph(
                root / "initial", root / "contracted", np.array([0, 0, 2]),
                free_disk_reserve_gib=0)
            direct = initial_graph(
                source, root / "direct", np.array([0, 0, 0, 0, 2]), np.arange(5),
                "absolute", "bacteria", free_disk_reserve_gib=0)
            observed = self._edges(root / "contracted")
            expected = self._edges(root / "direct")
            np.testing.assert_array_equal(
                observed[["feature_j", "feature_k"]], expected[["feature_j", "feature_k"]])
            np.testing.assert_allclose(observed["numerator"], expected["numerator"])
            np.testing.assert_allclose(observed["weight"], expected["weight"])
            self.assertLess(contracted["vertices"], initial["vertices"])
            self.assertLessEqual(contracted["edge_count"], initial["edge_count"])
            self.assertTrue(np.any(observed["feature_j"] == observed["feature_k"]))

    def test_gene_projection_clusters_graph_without_bacteria_context(self):
        pair_gene = np.array([0, 0, 1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_graph(
                self._pair_graph(root), root / "gene", pair_gene, np.arange(5),
                "absolute", "gene", bucket_count=4, free_disk_reserve_gib=0)
            worker = Path(__file__).resolve().parents[1] / "scripts/bernoulli_variation_cluster_worker.py"
            result = subprocess.run([
                str(JOIN_PYTHON), str(worker), "coprojection", str(root / "gene"),
                str(root / "community"), "--threshold", ".7", "--workers", "2",
                "--free-disk-gb", "0", "--memory-gb", "1",
            ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            labels = aligned_labels(root / "community/clusters.csv.gz", np.array([0, 1]))
            self.assertEqual(len(labels), 2)

    def test_canonical_labels_follow_smallest_entity(self):
        observed = canonical_labels(np.array([20, 5, 9, 30]), np.array([7, 4, 7, 4]))
        self.assertEqual(observed.tolist(), [1, 0, 1, 0])

    def test_coprojection_uses_summed_sqrt_marginals(self):
        matrix = sparse.csr_matrix([[.1, .2, .7], [.4, .1, .5]])
        train, test, sizes = _clustered_marginals(
            matrix, matrix, np.array([0, 1, 2]), np.array([0, 1, 2]),
            np.array([0, 0, 1]))
        expected = np.sqrt([[.3, .7], [.5, .5]])
        np.testing.assert_allclose(train.toarray(), expected)
        np.testing.assert_allclose(test.toarray(), expected)
        self.assertEqual(sizes.tolist(), [2, 1])

    def test_contracted_ids_compose_deterministically(self):
        first = contracted_ids(np.array([4, 1, 8]), np.array([7, 7, 2]))
        self.assertEqual(first.tolist(), [1, 1, 8])
        second = contracted_ids(np.unique(first), np.array([3, 3]))
        mapping = dict(zip(np.unique(first), second))
        self.assertEqual([mapping[value] for value in first], [1, 1, 1])

    def test_uses_stored_effective_graph_without_rejoining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "real/D/bernoulli_join_graph/graph"
            effective = source.parent / "graph-threshold-0.70"
            for path, threshold in ((source, .6), (effective, .7)):
                (path / "edges").mkdir(parents=True)
                (path / "run.json").write_text(json.dumps({"threshold": threshold}))
            with patch("scripts.run_bernoulli_variations.OUTPUT_ROOT", root):
                self.assertEqual(graph_path("D", {"effective_threshold": .7}), effective)
                self.assertEqual(graph_path("D", {"effective_threshold": .6}), source)

    def test_kaminpar_integer_weight_guard(self):
        code = ("from community_detection import safe_weight_scale; "
                "assert safe_weight_scale(500_000_000, 4) == 2; "
                "assert safe_weight_scale(100, 4) == 4; "
                "\ntry: safe_weight_scale(1_100_000_000, 4)\n"
                "except RuntimeError: pass\nelse: raise AssertionError('missing overflow guard')")
        subprocess.run([str(JOIN_PYTHON), "-c", code], check=True,
                       cwd=Path("/home/kapachy/Microbium/bernoulli-correlation-join"),
                       capture_output=True)

    def test_variation_worker_uses_200_gib_without_available_memory_preflight(self):
        worker = (Path(__file__).resolve().parents[1]
                  / "scripts/bernoulli_variation_cluster_worker.py")
        source = worker.read_text()
        self.assertIn('default=200', source)
        self.assertNotIn('insufficient available memory', source)

    def test_external_signed_leiden_and_kaminpar_split_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graph"
            (graph / "edges").mkdir(parents=True)
            nodes = np.zeros(5, dtype=[
                ("feature_id", "<u4"), ("column_index", "<u4"),
                ("support", "<u2"), ("minor_support", "<u2"),
                ("flipped", "?"), ("constant", "?"), ("eligible", "?"),
                ("exclusion_code", "u1"),
            ])
            nodes["feature_id"] = [1, 2, 3, 4, 5]
            nodes["column_index"] = np.arange(5)
            nodes["eligible"] = True
            np.save(graph / "nodes.npy", nodes)
            edges = np.array([(1, 2, .9), (3, 4, -.8)], dtype=[
                ("feature_j", "<u4"), ("feature_k", "<u4"), ("phi", "<f4")])
            np.save(graph / "edges/part-000000.npy", edges)
            (graph / "run.json").write_text(json.dumps({"status": "complete", "threshold": .7,
                "edge_count": 2, "edge_weight": {"field": "phi", "transform": "abs"}}))
            worker = Path(__file__).resolve().parents[1] / "scripts/bernoulli_variation_cluster_worker.py"
            for variation, extra in (("signed_families", []), ("kaminpar", ["--k", "1"])):
                result = subprocess.run([str(JOIN_PYTHON), str(worker), variation,
                    str(graph), str(root / variation), "--threshold", ".7", "--workers", "2",
                    "--free-disk-gb", "0", "--memory-gb", "1", *extra],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            positive = pd.read_csv(root / "signed_families/positive_community/clusters.csv.gz")
            negative = pd.read_csv(root / "signed_families/negative_community/clusters.csv.gz")
            positive_summary = json.loads(
                (root / "signed_families/positive_community/community.json").read_text())
            negative_summary = json.loads(
                (root / "signed_families/negative_community/community.json").read_text())
            self.assertEqual((positive_summary["edge_sign"], positive_summary["leiden_iterations"]), (1, 2))
            self.assertEqual((negative_summary["edge_sign"], negative_summary["leiden_iterations"]), (0, 2))
            self.assertFalse((root / "signed_families/positive_graph").exists())
            self.assertEqual(positive.loc[positive.feature_id == 1, "cluster"].iloc[0],
                             positive.loc[positive.feature_id == 2, "cluster"].iloc[0])
            self.assertEqual(negative.loc[negative.feature_id == 3, "cluster"].iloc[0],
                             negative.loc[negative.feature_id == 4, "cluster"].iloc[0])
            labels = aligned_labels(root / "kaminpar/community/clusters.csv.gz", np.arange(1, 6))
            self.assertEqual(len(np.unique(labels)), 3)
            self.assertEqual(labels[0], labels[1])
            self.assertEqual(labels[2], labels[3])
            self.assertNotEqual(labels[0], labels[2])

    def test_representative_is_training_variance_then_pair_id(self):
        train = sparse.csr_matrix([[0, 1, 0, 1], [1, 0, 0, 0], [0, 1, 2, 0]], dtype=float)
        labels = np.array([1, 1, 0, 0])
        chosen, variance = choose_representatives(train, labels, np.array([20, 10, 40, 30]))
        self.assertEqual(chosen.tolist(), [2, 1])
        self.assertGreater(variance[2], variance[3])
        self.assertEqual(variance[0], variance[1])

    def test_bacterium_split_only_splits_baseline_clusters(self):
        groups, labels = split_by_bacterium(np.array([0, 0, 1, 1]), np.array([2, 3, 2, 2]))
        self.assertEqual(groups.tolist(), [[0, 2], [0, 3], [1, 2]])
        self.assertEqual(labels.tolist(), [0, 1, 2, 2])

    def test_signed_filter_and_oriented_contrast(self):
        dtype = [("feature_j", "<u4"), ("feature_k", "<u4"), ("phi", "<f4")]
        edges = np.array([(20, 10, -0.8), (10, 30, 0.9), (20, 30, -0.7)], dtype=dtype)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "edges").mkdir(parents=True)
            np.save(source / "nodes.npy", np.arange(3, dtype=np.uint32))
            np.save(source / "edges/part-000000.npy", edges)
            (source / "run.json").write_text(json.dumps({"status": "complete", "threshold": 0.6,
                "edge_count": 3, "edge_weight": {"field": "phi", "transform": "abs"}}))
            code = ("from pathlib import Path; from scripts.bernoulli_variation_cluster_worker import signed_graph; "
                    f"s=Path({str(source)!r}); signed_graph(s,s.parent/'pos',1,0.7,0); "
                    "signed_graph(s,s.parent/'neg',-1,0.7,0)")
            subprocess.run([str(JOIN_PYTHON), "-c", code], check=True,
                           cwd=Path(__file__).resolve().parents[1], capture_output=True)
            pos = np.load(root / "pos/edges/part-000000.npy")
            neg = np.load(root / "neg/edges/part-000000.npy")
            self.assertEqual(pos.size, 1)
            self.assertEqual(neg.size, 2)
            self.assertTrue(np.all(pos["phi"] > 0))
            self.assertTrue(np.all(neg["phi"] < 0))
            ids = np.array([10, 20, 30])
            endpoints = strongest_negative_edges(root / "neg", ids, np.array([0, 0, 0]))
            self.assertEqual(endpoints[0][:3], (0, 10, 20))
            train = sparse.csr_matrix([[0.25, 1.0, 0.0], [1.0, 0.25, 0.0]])
            result, _, inventory = signed_features(train, train, ids, np.array([0, 0, 1]),
                                                    np.array([0, 0, 0]), root / "neg")
            np.testing.assert_allclose(result.toarray()[:, -1],
                                       [0.5 - 1.0, 1.0 - 0.5])
            self.assertEqual(inventory.iloc[-1]["family"], "negative_contrast")

    def test_unassigned_constant_columns_remain_singletons(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.csv.gz"
            pd.DataFrame({"feature_id": [2, 3, 5], "cluster": [4, -1, -1]}).to_csv(path, index=False)
            labels = aligned_labels(path, np.array([2, 3, 5]))
            self.assertEqual(len(np.unique(labels)), 3)

    def test_dataset_summary_uses_coarse_k_only(self):
        data = pd.DataFrame({"dataset": ["A", "A", "A", "A"],
                             "method": ["baseline", "baseline", "max_variance", "max_variance"],
                             "requested_k": [1, 5, 1, 5], "effective_k": [1, 5, 1, 5],
                             "auroc": [1, .6, 0, .7], "precision": [.8] * 4,
                             "recall": [.8] * 4, "f1": [.8] * 4})
        dataset, _ = summarize_rows(data)
        observed = dataset.set_index("method")
        self.assertAlmostEqual(observed.loc["baseline", "mean_auroc"], .6)
        self.assertAlmostEqual(observed.loc["max_variance", "delta_mean_auroc_vs_baseline"], .1)
        duplicated = pd.concat([data, data.iloc[[1]].assign(auroc=0.0)])
        revised, _ = summarize_rows(duplicated)
        self.assertAlmostEqual(revised.set_index("method").loc["baseline", "mean_auroc"], .6)


if __name__ == "__main__":
    unittest.main()
