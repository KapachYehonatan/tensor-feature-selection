import tempfile
import unittest
from pathlib import Path
import json
import subprocess
from unittest.mock import patch

import numpy as np
from scipy import sparse

from scripts.bernoulli_join_worker import (
    MAX_COMMUNITY_EDGES,
    choose_effective_threshold,
    count_threshold_edges,
    fallback_thresholds,
    pack_sparse_detection,
    stable_cluster_labels,
)
from scripts.run_bernoulli_correlation_join import aggregate_pair_clusters, JOIN_PYTHON, signature
from scripts.run_method_comparison import METHODS
from scripts.summarize_method_comparison import METHOD_ORDER

PROJECT = Path(__file__).resolve().parents[1]


class BernoulliJoinMethodTests(unittest.TestCase):
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
            ("feature_j", "<u4"), ("feature_k", "<u4"), ("jaccard", "<f4")
        ])
        edges["jaccard"] = [np.float32(0.9), np.nextafter(np.float32(0.9), 0)]
        manifest = {"edge_weight": {"field": "jaccard", "transform": "identity"}}
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
                str(PROJECT / "scripts/bernoulli_join_worker.py"),
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
                str(PROJECT / "scripts/bernoulli_join_worker.py"),
                str(matrix_path), str(split_path), str(output),
                "--threshold", "0.7", "--workers", "2",
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

    def test_external_jaccard_worker_matches_brute_force(self):
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
                str(PROJECT / "scripts/bernoulli_join_worker.py"),
                str(matrix_path), str(split_path), str(output),
                "--join-method", "jaccard", "--threshold", "0.6",
                "--workers", "2", "--memory-gb", "1",
                "--working-memory-gb", "1",
                "--community-working-memory-gb", "1",
                "--output-gb", "1", "--free-disk-gb", "0",
                "--memory-reserve-gb", "0", "--keep-graph-gb", "1",
            ], check=True, capture_output=True, text=True)
            observed = set()
            for shard_path in sorted((output / "graph/edges").glob("part-*.npy")):
                for row in np.load(shard_path):
                    observed.add(tuple(sorted((int(row["feature_j"]), int(row["feature_k"])))))
            integration = json.loads(
                (output / "integration.json").read_text(encoding="utf-8")
            )
            with np.load(output / "assignments.npz") as saved:
                labels = saved["clusters"]

        expected = set()
        for left in range(4):
            for right in range(left + 1, 4):
                intersection = np.sum(detections[:, left] & detections[:, right])
                union = np.sum(detections[:, left] | detections[:, right])
                if intersection / union >= 0.6:
                    expected.add((left, right))
        self.assertEqual(observed, expected)
        self.assertEqual(integration["join_method"], "jaccard")
        self.assertNotIn(labels[4], labels[:4])

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

    def test_original_abundances_are_mean_aggregated_then_sqrt_transformed(self):
        train = sparse.csr_matrix([[0.2, 0.4, 0.4], [0.0, 0.6, 0.4]])
        test = sparse.csr_matrix([[0.1, 0.3, 0.6]])
        observed_train, observed_test, sizes = aggregate_pair_clusters(
            train, test, np.array([0, 0, 1])
        )
        np.testing.assert_allclose(
            observed_train.toarray(), np.sqrt([[0.3, 0.4], [0.3, 0.4]])
        )
        np.testing.assert_allclose(observed_test.toarray(), np.sqrt([[0.2, 0.6]]))
        np.testing.assert_array_equal(sizes, [2, 1])

    def test_new_reports_expose_only_requested_methods(self):
        self.assertEqual(
            METHOD_ORDER,
            [
                "bh_pairs_bacteria",
                "bh_bacteria",
                "bh_genes",
                "clustered_plus_pairs",
                "bernoulli_correlation_join",
                "jaccard_join",
            ],
        )
        self.assertNotIn("bh_pairs", METHODS)
        self.assertNotIn("clustered_marginals", METHODS)

    def test_graph_cache_survives_evaluation_grid_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "X_valid.npz").write_bytes(b"matrix-fingerprint")
            (source / "samples.csv").write_text("y\n0\n1\n", encoding="utf-8")
            job = {"input": str(source), "dataset": "tiny"}
            y = np.array([0, 1])
            graph_before = signature(job, y, graph_only=True)
            method_before = signature(job, y)
            with patch(
                "scripts.run_bernoulli_correlation_join.K_VALUES", [1, 2, 3]
            ):
                graph_after = signature(job, y, graph_only=True)
                method_after = signature(job, y)
        self.assertEqual(graph_before, graph_after)
        self.assertNotEqual(method_before, method_after)


if __name__ == "__main__":
    unittest.main()
