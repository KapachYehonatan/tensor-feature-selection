import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from research_utils import (  # noqa: E402
    _column_similarity,
    benjamini_hochberg,
    contextual_profiles,
    exhaustive_similarity_clusters,
    grouped_residual_features,
    normalize_rows,
    prediction_matrix,
    prevalence_mask,
    similarity_clusters,
    unique_day_zero_stool_mask,
)
from run_datasets import MAX_PARALLEL_DATASETS, N_CPUS, cpu_allocation, datasets_to_run
from feature_selection.run_feature_selection import (
    pair_bh_features,
    run_holdout_selection,
    selected_columns,
)
from scripts.run_method_comparison import (
    K_VALUES,
    MAX_CLUSTER_ROUNDS,
    bh_family,
    collapsed_average,
    evaluate_ranked,
    is_complete,
    rsp_row,
    synthetic_jobs,
    write_completion,
)
from scripts.summarize_method_comparison import curve_summary, seed_average
from scripts.generate_synthetic import (  # noqa: E402
    generate_size,
    load_configuration,
    make_block_prevalence,
    random_streams,
    synthetic_worker_count,
    tensor_to_csr,
)


class ResearchStepTests(unittest.TestCase):
    def test_comparison_budget_and_clustering_rounds(self):
        self.assertEqual(K_VALUES, [5, 10, 20, 35, 50, 60, 75, 85, 100])
        self.assertEqual(MAX_CLUSTER_ROUNDS, 4)

    def test_marginal_and_pair_selection_have_no_200_feature_cap(self):
        y = np.repeat([0, 1], 10)
        values = np.tile(y[:, None], (1, 225))
        selected, _ = selected_columns(sparse.csr_matrix(values), y)
        self.assertEqual(len(selected), 225)

        scores = {
            "p": np.zeros(225), "q": np.zeros(225), "effect": np.ones(225)
        }
        with patch(
            "feature_selection.run_feature_selection.pair_bh_scores",
            return_value=scores,
        ):
            result = pair_bh_features(
                sparse.csr_matrix((20, 225)), sparse.csr_matrix((4, 225)),
                np.arange(225), sparse.csr_matrix((20, 0)), y,
            )
        self.assertEqual(
            (result["train"].shape[1], result["test"].shape[1], len(result["selected"])),
            (225, 225, 225),
        )

    def test_comparison_collapsed_average_and_feature_counts(self):
        matrix = sparse.csr_matrix([[2.0, 4.0, 3.0], [0.0, 2.0, 6.0]])
        collapsed, active = collapsed_average(matrix, np.array([0, 0, 1]), 3)
        np.testing.assert_array_equal(active, [0, 1])
        np.testing.assert_allclose(collapsed.toarray(), np.sqrt([[3.0, 3.0], [1.0, 6.0]]))

        ranking = pd.DataFrame({"level": ["bacterium", "gene", "pair"]})
        with patch(
            "scripts.run_method_comparison.predict_model",
            return_value=np.array([0.2, 0.8, 0.7, 0.3]),
        ):
            metrics, _ = evaluate_ranked(
                "clustered_plus_pairs",
                sparse.csr_matrix(np.ones((8, 3))),
                sparse.csr_matrix(np.ones((4, 3))),
                ranking,
                np.tile([0, 1], 4), np.array([0, 1, 1, 0]),
                {"bacteria": 1, "gene": 1, "pair": 1},
            )
        last = metrics.iloc[-1]
        self.assertEqual(
            tuple(last[["bacteria_features", "gene_features", "pair_features", "effective_k"]]),
            (1, 1, 1, 3),
        )

    def test_comparison_bh_rsp_and_cache_are_deterministic(self):
        matrix = sparse.csr_matrix([
            [0, 0, 1], [0, 1, 0], [0, 0, 2], [1, 1, 0],
            [4, 0, 1], [3, 1, 0], [5, 0, 2], [4, 1, 0],
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        first = bh_family(matrix, [y, y[::-1]])
        second = bh_family(matrix, [y, y[::-1]])
        np.testing.assert_array_equal(first["order"], second["order"])
        np.testing.assert_allclose(first["q"], second["q"])
        score = rsp_row("bh_pairs", [20, 1, 3, 2, 2, 2])
        self.assertAlmostEqual(score["sp"], 2.0)
        self.assertAlmostEqual(score["rsp"], -1 / 3)
        self.assertEqual(rsp_row("bh_pairs", [0, 0, 0, 0, 0, 0])["rsp"], 0)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            write_completion(cache, "abc")
            self.assertTrue(is_complete(cache, "abc"))
            self.assertFalse(is_complete(cache, "different"))

    def test_pair_test_returns_direct_bh_values(self):
        rng = np.random.default_rng(1)
        y = np.repeat([0, 1], 30)
        values = np.column_stack([
            3 * y,
            rng.binomial(1, 0.5, len(y)),
            rng.binomial(1, 0.5, len(y)),
            rng.binomial(1, 0.5, len(y)),
        ]).astype(float)
        result = pair_bh_features(
            sparse.csr_matrix(values), sparse.csr_matrix(values[:10]),
            np.array([0, 0, 1, 1]), sparse.csr_matrix((len(y), 0)), y,
        )
        np.testing.assert_allclose(result["q"], benjamini_hochberg(result["p"]))
        self.assertIn(0, result["selected"])
        self.assertNotIn(2, result["selected"])

    def test_comparison_uses_only_full_signal_synthetic_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data" / "main" / "n_12" / "seed_7" / "full_signals"
            dataset.mkdir(parents=True)
            np.save(dataset / "target.npy", np.array([0, 1]))
            configuration = {"paths": {"data_root": "data"}}
            preset = {"sample_sizes": [12], "seeds": [7]}
            with patch("scripts.run_method_comparison.PROJECT", root), patch(
                "scripts.run_method_comparison.load_configuration",
                return_value=(configuration, "main", preset),
            ):
                jobs = synthetic_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["scenario"], "full_signals")

    def test_comparison_seed_average_precedes_n_summary(self):
        rows = []
        for seed, score in [(1, 0.6), (2, 0.8)]:
            for k in [50, 100]:
                rows.append({
                    "source": "synthetic", "dataset": "n_20", "n": 20,
                    "seed": seed, "scenario": "full_signals", "method": "bh_pairs",
                    "method_label": "BH", "requested_k": k, "effective_k": k,
                    "auroc": score, "precision": score, "recall": score, "f1": score,
                    "output_bacteria_features": 0, "output_gene_features": 0,
                    "output_pair_features": 100, "output_total_features": 100,
                    "selected_bacteria_features": 0, "selected_gene_features": 0,
                    "selected_pair_features": 4, "selected_total_features": 4,
                })
        averaged = seed_average(pd.DataFrame(rows))
        self.assertTrue(np.allclose(averaged["auroc"], 0.7))
        summary = curve_summary(
            averaged, ["source", "dataset", "n", "method", "method_label"]
        )
        self.assertAlmostEqual(summary.loc[0, "mean_auroc"], 0.7)

    def test_synthetic_worker_count_is_configurable_and_capped(self):
        with patch.dict(os.environ, {"SYNTHETIC_WORKERS": "10"}):
            self.assertEqual(synthetic_worker_count(6), 6)
        with patch.dict(os.environ, {"SYNTHETIC_WORKERS": "0"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                synthetic_worker_count(6)

    def test_pair_only_model_is_conditional(self):
        matrix = sparse.csr_matrix(np.ones((20, 1)))
        y = np.tile([0, 1], 10)

        def representations(x_train, x_test, *_args, **_kwargs):
            train = sparse.csr_matrix(np.ones((x_train.shape[0], 1)))
            test = sparse.csr_matrix(np.ones((x_test.shape[0], 1)))
            return {
                "bacteria_clusters": (train, test),
                "gene_clusters": (train, test),
                "marginal_clusters": (
                    sparse.hstack([train, train], format="csr"),
                    sparse.hstack([test, test], format="csr"),
                ),
            }, {
                "threshold": 1,
                "pair_mask": np.array([True]),
                "pair_train": train,
                "pair_test": test,
                "pair_blocks": np.array([0]),
                "active_bacteria": np.array([0]),
                "active_genes": np.array([0]),
                "bacteria_labels": np.array([0]),
                "gene_labels": np.array([0]),
                "selected_bacteria_clusters": np.array([0]),
                "selected_gene_clusters": np.array([0]),
            }

        def prediction(_train_x, _train_y, test_x, _seed):
            return np.full(test_x.shape[0], 0.5)

        for selected_count in (0, 1):
            selected = np.arange(selected_count)

            def pairs(pair_train, pair_test, *_args, **_kwargs):
                return {
                    "train": pair_train[:, :selected_count],
                    "test": pair_test[:, :selected_count],
                    "selected": selected,
                    "p": np.ones(pair_train.shape[1]),
                    "q": np.ones(pair_train.shape[1]),
                    "effect": np.zeros(pair_train.shape[1]),
                }

            with patch(
                "feature_selection.run_feature_selection.holdout_representations",
                side_effect=representations,
            ), patch(
                "feature_selection.run_feature_selection.pair_bh_features",
                side_effect=pairs,
            ), patch(
                "feature_selection.run_feature_selection.predict_model",
                side_effect=prediction,
            ):
                result = run_holdout_selection(
                    matrix, y, np.array([0]), np.array([0])
                )

            modes = set(result["metrics"]["mode"])
            self.assertNotIn("unclustered_marginals", modes)
            self.assertEqual("pairs_only" in modes, bool(selected_count))
            if selected_count:
                row = result["fold_model_info"].set_index("mode").loc["pairs_only"]
                self.assertEqual(
                    tuple(row[["bacteria_features", "gene_features", "pair_features"]]),
                    (0, 0, 1),
                )

    def test_normalization_rejects_empty_samples(self):
        with self.assertRaisesRegex(ValueError, "1 samples"):
            normalize_rows(sparse.csr_matrix([[0, 0], [2, 2]]))

    def test_dataset_runner_accepts_all_one_or_list(self):
        all_datasets = datasets_to_run("all")
        self.assertIn("WirbelJ_2018", all_datasets)
        self.assertIn("HMP_2019_ibdmdb_unique", all_datasets)
        self.assertIn("NielsenHB_2014_unique", all_datasets)
        self.assertNotIn("BrooksB_2017", all_datasets)
        self.assertEqual(datasets_to_run("WirbelJ_2018"), ["WirbelJ_2018"])
        self.assertEqual(
            datasets_to_run(["WirbelJ_2018", "QinJ_2012"]),
            ["WirbelJ_2018", "QinJ_2012"],
        )

    def test_unique_baseline_prefers_read_depth(self):
        selected = unique_day_zero_stool_mask(
            ["A", "A", "B", "C", "D"],
            ["stool", "stool", "stool", "oralcavity", "stool"],
            [0, 0, 1, 0, 0],
            [10, 20, 30, 40, np.nan],
            ["a1", "a2", "b1", "c1", "d1"],
        )
        np.testing.assert_array_equal(selected, [False, True, False, False, True])

    def test_dataset_runner_cpu_allocation(self):
        workers, threads, cpus = cpu_allocation(6, range(N_CPUS + 4))
        expected_workers = min(MAX_PARALLEL_DATASETS, N_CPUS, 6)
        self.assertEqual(workers, expected_workers)
        self.assertEqual(threads, N_CPUS // expected_workers)
        self.assertEqual(cpus, list(range(N_CPUS)))

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
                expected, qualifying = exhaustive_similarity_clusters(
                    profiles, similarity_threshold, 0.70, metric
                )
                observed, diagnostics = similarity_clusters(
                    profiles,
                    similarity_threshold,
                    0.70,
                    metric,
                    return_diagnostics=True,
                    collect_checked_pairs=True,
                )
                np.testing.assert_array_equal(observed, expected)
                self.assertTrue(
                    set(qualifying).issubset(set(diagnostics["checked_pairs"]))
                )
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

    def test_single_target_predictions_remain_a_column(self):
        prediction = prediction_matrix(np.array([0.2, 0.8]), 2)
        np.testing.assert_array_equal(prediction, [[0.2], [0.8]])
        self.assertEqual(prediction_matrix(np.ones((2, 3)), 2).shape, (2, 3))

    def test_synthetic_configuration_has_smoke_and_main_scales(self):
        configuration, name, smoke = load_configuration(preset_name="smoke")
        self.assertEqual(name, "smoke")
        self.assertEqual(smoke["sample_sizes"], [40])
        main = configuration["presets"]["main"]
        self.assertEqual(main["sample_sizes"], [150, 250, 350, 500])
        self.assertEqual(len(main["seeds"]), 7)
        self.assertEqual(
            (main["bacteria"], main["genes"], main["bacteria_clusters"], main["gene_clusters"]),
            (50, 10_000, 10, 100),
        )
        self.assertEqual(main["pair_count"], 4)
        self.assertEqual(main["pair_effect"], 5.0)
        self.assertNotIn("pair_detection_low", main)
        _, rng = random_streams(101, 40)
        prevalence = make_block_prevalence(smoke, rng["prevalence"])
        self.assertEqual(prevalence.min(), 0.01)
        self.assertEqual(prevalence.max(), 0.40)
        self.assertAlmostEqual(prevalence.mean(), 0.097)

    def test_synthetic_generation_is_reproducible_and_sparse(self):
        configuration, _, preset = load_configuration(preset_name="smoke")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            roots = []
            for directory in (first, second):
                local = copy.deepcopy(configuration)
                local["paths"]["data_root"] = directory
                generate_size(local, "smoke", preset, 40, 101)
                roots.append(Path(directory) / "smoke" / "n_40" / "seed_101")

            for scenario in configuration["scenarios"]:
                first_tensor = np.load(roots[0] / scenario / "tensor.npz")
                second_tensor = np.load(roots[1] / scenario / "tensor.npz")
                for key in first_tensor.files:
                    np.testing.assert_array_equal(first_tensor[key], second_tensor[key])

            noisy = roots[0] / "noisy_blocks"
            matrix, edge_bacteria, edge_gene, shape = tensor_to_csr(noisy)
            self.assertEqual(shape, (40, 10, 200))
            self.assertEqual(len(edge_bacteria), 2000)
            self.assertEqual(len(edge_gene), 2000)
            sparsity = 1 - matrix.nnz / np.prod(shape)
            self.assertGreaterEqual(sparsity, 0.88)
            self.assertLessEqual(sparsity, 0.93)
            pd.testing.assert_frame_equal(
                pd.read_csv(roots[0] / "exact_blocks" / "block_truth.csv.gz"),
                pd.read_csv(noisy / "block_truth.csv.gz"),
            )

            noisy_inode = (noisy / "tensor.npz").stat().st_ino
            for scenario in (
                "null_target", "marginal_signals", "pair_signals", "full_signals"
            ):
                self.assertEqual(
                    (roots[0] / scenario / "tensor.npz").stat().st_ino,
                    noisy_inode,
                )
                with np.load(roots[0] / scenario / "target_model.npz") as model:
                    np.testing.assert_allclose(
                        model["probability"], expit(model["linear_predictor"])
                    )
                    self.assertAlmostEqual(float(model["probability"].mean()), 0.5)
            with np.load(roots[0] / "null_target" / "target_model.npz") as model:
                np.testing.assert_allclose(model["probability"], 0.5)

    def test_pair_truth_uses_distinct_nonmarginal_entities(self):
        configuration, _, preset = load_configuration(preset_name="smoke")
        with tempfile.TemporaryDirectory() as directory:
            local = copy.deepcopy(configuration)
            local["paths"]["data_root"] = directory
            generate_size(local, "smoke", preset, 40, 101)
            root = Path(directory) / "smoke" / "n_40" / "seed_101"
            signals = pd.read_csv(root / "full_signals" / "signal_truth.csv.gz")
            pairs = signals[signals["level"].eq("pair")]
            bacteria = pd.read_csv(root / "full_signals" / "bacteria_truth.csv.gz")
            genes = pd.read_csv(root / "full_signals" / "gene_truth.csv.gz")
            self.assertEqual(len(pairs), preset["pair_count"])
            self.assertEqual(pairs["bacterium"].nunique(), preset["pair_count"])
            self.assertEqual(pairs["gene"].nunique(), preset["pair_count"])
            self.assertFalse(
                bacteria.loc[pairs["bacterium"].astype(int), "true_cluster"].isin([0]).any()
            )
            self.assertFalse(
                genes.loc[pairs["gene"].astype(int), "true_cluster"].isin([1, 7]).any()
            )
            self.assertGreaterEqual(pairs["expected_prevalence"].min(), 0.38)
            self.assertGreaterEqual(pairs["realized_support"].min(), 5)
            self.assertEqual(set(pairs["direction"]), {"positive", "negative"})

    def test_bh_is_monotone_in_rank(self):
        p = np.array([0.03, 0.001, 0.02, 0.8])
        q = benjamini_hochberg(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= 0))
        self.assertTrue(np.all((0 <= q) & (q <= 1)))

    def test_grouped_pair_residuals_remove_block_aggregate(self):
        matrix = sparse.csr_matrix([
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 0.0, 1.0, 2.0],
        ])
        groups = np.array([0, 0, 1, 1])
        residual, _ = grouped_residual_features(
            matrix, matrix, groups, np.arange(matrix.shape[1])
        )
        totals = np.column_stack([
            np.asarray(matrix[:, groups == group].sum(axis=1)).ravel()
            for group in np.unique(groups)
        ])
        centered_totals = totals - totals.mean(axis=0)
        for column, group in enumerate(groups):
            self.assertAlmostEqual(
                float(residual[:, column].toarray().ravel() @ centered_totals[:, group]),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
