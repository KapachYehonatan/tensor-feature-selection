import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from research_utils import (  # noqa: E402
    _column_similarity,
    benjamini_hochberg,
    contextual_profiles,
    normalize_rows,
    prevalence_mask,
    similarity_clusters,
)
from run_datasets import cpu_allocation, datasets_to_run


def brute_representative_labels(profiles, similarity_threshold, jaccard_threshold, metric):
    """Small exhaustive reference for the representative clustering rule."""
    csc = sparse.csc_matrix(profiles, dtype=float)
    csc.eliminate_zeros()
    csc.sort_indices()
    support = np.diff(csc.indptr)
    labels = np.full(csc.shape[1], -1, dtype=np.int64)
    representatives = []
    feature_order = np.lexsort((np.arange(csc.shape[1]), support))
    for feature in feature_order:
        best_representative = None
        best_key = None
        for representative in representatives:
            similarity, jaccard = _column_similarity(csc, feature, representative, metric)
            if similarity < similarity_threshold or jaccard < jaccard_threshold:
                continue
            key = (similarity, jaccard, -representative)
            if best_key is None or key > best_key:
                best_key = key
                best_representative = representative
        if best_representative is None:
            labels[feature] = len(representatives)
            representatives.append(feature)
        else:
            labels[feature] = labels[best_representative]
    return labels


class ResearchStepTests(unittest.TestCase):
    def test_normalization_rejects_empty_samples(self):
        with self.assertRaisesRegex(ValueError, "1 samples"):
            normalize_rows(sparse.csr_matrix([[0, 0], [2, 2]]))

    def test_dataset_runner_accepts_all_one_or_list(self):
        all_datasets = datasets_to_run("all")
        self.assertIn("WirbelJ_2018", all_datasets)
        self.assertNotIn("BrooksB_2017", all_datasets)
        self.assertEqual(datasets_to_run("WirbelJ_2018"), ["WirbelJ_2018"])
        self.assertEqual(
            datasets_to_run(["WirbelJ_2018", "QinJ_2012"]),
            ["WirbelJ_2018", "QinJ_2012"],
        )

    def test_dataset_runner_cpu_allocation(self):
        workers, threads, cpus = cpu_allocation(6, range(20))
        self.assertEqual(workers, 3)
        self.assertEqual(threads, 4)
        self.assertEqual(cpus, list(range(12)))

    def test_prevalence_counts_samples(self):
        matrix = sparse.csr_matrix(
            [
                [1, 1, 0],
                [2, 0, 0],
                [0, 1, 1],
                [0, 0, 1],
            ]
        )
        mask, support, threshold = prevalence_mask(matrix, fraction=0.5, minimum_samples=1)
        np.testing.assert_array_equal(support, [2, 2, 2])
        np.testing.assert_array_equal(mask, [True, True, True])
        self.assertEqual(threshold, 2)

    def test_similarity_threshold_finds_redundant_features_without_k(self):
        profiles = sparse.csr_matrix(
            np.array(
                [
                    [1.0, 2.0, 0.0],
                    [2.0, 4.0, 1.0],
                    [0.0, 0.0, 2.0],
                    [4.0, 8.0, 0.0],
                ]
            )
        )
        labels = similarity_clusters(
            profiles, similarity_threshold=0.95, jaccard_threshold=0.80
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_sparse_join_matches_exhaustive_representative_search(self):
        for metric, similarity_threshold in (("correlation", 0.60), ("bray_curtis", 0.55)):
            for seed in range(10):
                rng = np.random.default_rng(seed)
                profiles = rng.integers(0, 5, size=(30, 40)).astype(float)
                profiles[rng.random(profiles.shape) < 0.65] = 0
                for feature in range(profiles.shape[1]):
                    if not profiles[:, feature].any():
                        profiles[feature % len(profiles), feature] = 1
                expected = brute_representative_labels(
                    profiles, similarity_threshold, 0.70, metric
                )
                observed = similarity_clusters(
                    profiles, similarity_threshold, 0.70, metric
                )
                np.testing.assert_array_equal(observed, expected)
                csc = sparse.csc_matrix(profiles)
                support = np.diff(csc.indptr)
                for label in np.unique(observed):
                    members = np.flatnonzero(observed == label)
                    representative = min(members, key=lambda member: (support[member], member))
                    for member in members[members != representative]:
                        similarity, jaccard = _column_similarity(
                            csc, member, representative, metric
                        )
                        self.assertGreaterEqual(similarity, similarity_threshold)
                        self.assertGreaterEqual(jaccard, 0.70)

    def test_near_detection_patterns_are_compared_exactly(self):
        profiles = np.zeros((15, 3))
        profiles[:10, 0] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0.01]
        profiles[:9, 1] = np.arange(1, 10)
        profiles[5:, 2] = 1
        labels, diagnostics = similarity_clusters(
            profiles,
            similarity_threshold=0.95,
            jaccard_threshold=0.80,
            return_diagnostics=True,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])
        self.assertGreater(diagnostics["exact_jaccard_checks"], 0)
        self.assertEqual(diagnostics["accepted_assignments"], 1)
        np.testing.assert_array_equal(
            labels,
            similarity_clusters(profiles, 0.95, 0.80),
        )

    def test_jaccard_boundary_is_included(self):
        profiles = np.zeros((8, 2))
        profiles[:5, 0] = 1
        profiles[:4, 1] = 1
        labels = similarity_clusters(
            profiles,
            similarity_threshold=0.88,
            jaccard_threshold=0.80,
            metric="bray_curtis",
        )
        self.assertEqual(labels[0], labels[1])

    def test_similarity_clustering_rejects_zero_support(self):
        with self.assertRaisesRegex(ValueError, "1 similarity features"):
            similarity_clusters(sparse.csr_matrix([[1, 0], [2, 0]]))

    def test_context_profiles_preserve_mass(self):
        matrix = sparse.csr_matrix([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0]])
        profiles = contextual_profiles(
            matrix,
            pair_entity=np.array([0, 0, 1]),
            pair_context=np.array([0, 1, 1]),
            n_entities=2,
            n_contexts=2,
        )
        self.assertEqual(profiles.shape, (4, 2))
        self.assertAlmostEqual(float(profiles.sum()), float(matrix.sum()))

    def test_pair_cycle_preserves_both_margins(self):
        before = np.ones((2, 2))
        after = before + np.array([[0.25, -0.25], [-0.25, 0.25]])
        np.testing.assert_allclose(before.sum(axis=0), after.sum(axis=0))
        np.testing.assert_allclose(before.sum(axis=1), after.sum(axis=1))

    def test_bh_is_monotone_in_rank(self):
        p = np.array([0.03, 0.001, 0.02, 0.8])
        q = benjamini_hochberg(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= 0))
        self.assertTrue(np.all((0 <= q) & (q <= 1)))


if __name__ == "__main__":
    unittest.main()
