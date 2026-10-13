"""Small scientific invariants for the isolated QinJ side analysis."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy import sparse
from scripts.run_qinj_redundancy_analysis import select, residual_select, merge_features, diagnostics, construct, evaluate, validate_outer_split
import pandas as pd


class QinJRedundancyTests(unittest.TestCase):
    def test_legacy_outer_split_is_validated_not_assumed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            train, test, ids = np.array([2,0]), np.array([1]), np.array([4,7])
            with self.assertRaises(FileNotFoundError):
                validate_outer_split(base, train, test, ids)
            legacy = base / "bernoulli_join_split.npz"
            np.savez(legacy, train_index=train, test_index=test, feature_ids=ids)
            self.assertEqual(validate_outer_split(base, train, test, ids), legacy)
            with self.assertRaisesRegex(ValueError, "feature_ids"):
                validate_outer_split(base, train, test, ids[::-1])
            np.savez(base / "bernoulli_join_graph_split.npz", train_index=train[::-1], test_index=test, feature_ids=ids)
            with self.assertRaisesRegex(ValueError, "train_index"):
                validate_outer_split(base, train, test, ids)

    def test_filter_skips_duplicates_constants_and_refills(self):
        x = sparse.csr_matrix(np.array([[0, 0, 1, 0], [1, 2, 1, 1], [2, 4, 1, 0], [3, 6, 1, 1]], float))
        order, events = select(x, np.ones(4), "filter", .8)
        self.assertEqual(order, [0, 3])
        self.assertEqual(events[1][1:3], ("correlated", 0))
        self.assertEqual(events[2][1], "constant")
        self.assertEqual(select(x, np.ones(4), "filter", .8), (order, events))
        self.assertEqual(select(x[:, :0], [], "filter", .8)[0], [])

    def test_penalty_is_max_correlation_and_stable(self):
        x = sparse.csr_matrix([[0., 0, 0], [1, 2, 1], [2, 4, 0], [3, 6, 1]])
        self.assertEqual(select(x, np.array([.5, .49, .4]), "penalty", 0)[0], [0, 1, 2])
        self.assertEqual(select(x, np.array([.5, .49, .4]), "penalty", .5)[0], [0, 2, 1])

    def test_mean_penalty_is_average_not_maximum(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(40, 4)); a -= a.mean(axis=0)
        q, _ = np.linalg.qr(a)
        x = sparse.csr_matrix(np.column_stack([q[:, 0], q[:, 1],
            .7*q[:, 0] + np.sqrt(.51)*q[:, 2],
            .4*q[:, 0] + .4*q[:, 1] + np.sqrt(.68)*q[:, 3]]))
        effect = np.array([.9, .85, .7, .7])
        self.assertEqual(select(x, effect, "penalty", .5)[0][:3], [0, 1, 3])
        order, events = select(x, effect, "penalty_mean", .5)
        self.assertEqual(order[:3], [0, 1, 2])
        self.assertAlmostEqual(events[2][3], .35)

    def test_regression_penalties_match_direct_joint_ols(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 6))
        x[:, 1] = 2*x[:, 0]  # Rank-deficient selected sets must remain valid.
        x[:, 2] = x[:, 0] + x[:, 3] + .1*x[:, 2]
        x[:, 5] = 1
        target = (x[:, 0] + x[:, 4] > 0).astype(int)
        effect = np.array([.9, .8, .7, .6, .5, 9])
        for mode in ("penalty_r2", "penalty_residual_target"):
            order, events = select(sparse.csr_matrix(x), effect, mode, .5, target=target)
            self.assertEqual(order[0], 0)
            self.assertNotIn(5, order)
            chosen = []
            for i, _, _, penalty in events:
                penalties = {}
                for j in set(range(5)) - set(chosen):
                    if not chosen:
                        penalties[j] = 0
                        continue
                    design = np.column_stack([np.ones(len(x)), x[:, chosen]])
                    residual = x[:, j] - design @ np.linalg.lstsq(design, x[:, j], rcond=None)[0]
                    if mode == "penalty_r2":
                        penalties[j] = 1 - np.sum(residual**2)/np.sum((x[:, j]-x[:, j].mean())**2)
                    else:
                        r = abs(np.corrcoef(residual, target)[0, 1]) if np.linalg.norm(residual) > 1e-10 else 0
                        penalties[j] = 1 - r
                self.assertAlmostEqual(penalty, penalties[i], places=7)
                self.assertAlmostEqual(abs(effect[i])-.5*penalty,
                    max(abs(effect[j])-.5*v for j, v in penalties.items()), places=7)
                chosen.append(i)
            self.assertEqual(select(sparse.csr_matrix(x), effect, mode, .5, target=target)[0], order)
            self.assertEqual(select(sparse.csr_matrix(x), effect, mode, 0, target=target)[0], [0,1,2,3,4])
            self.assertEqual(select(sparse.csr_matrix((40,0)), [], mode, .5, target=target)[0], [])
        self.assertEqual(select(sparse.csr_matrix(x), effect, "penalty_residual_target", .5, target=1-target)[0], order)
        with self.assertRaises(ValueError):
            select(sparse.csr_matrix(x), effect, "penalty_residual_target", .5)
        with self.assertRaises(ValueError):
            select(sparse.csr_matrix(x), effect, "penalty_residual_target", .5, target=np.ones(40))

    def test_weighted_merge_uses_original_abundance(self):
        x = sparse.csr_matrix([[1., 2., 4.], [2., 4., 3.], [3., 6., 2.], [4., 8., 1.]])
        a, b, groups = merge_features(x, x * 2, np.array([1, 3, 2]), .9)
        self.assertEqual(groups, [[0, 1], [2]])
        np.testing.assert_allclose(a[:, 0].toarray().ravel(), np.sqrt((x[:, 0].toarray().ravel()**2 + 3*x[:, 1].toarray().ravel()**2)/4))
        np.testing.assert_allclose(b.toarray(), 2*a.toarray())

    def test_residual_selection_matches_joint_ols_and_ignores_later_pvalues(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 6))
        x[:, 1] = 3*x[:, 0]  # Exact duplicate with a different scale.
        x[:, 2] = x[:, 0] + .1*x[:, 2]
        x[:, 5] = 1  # Constant is not a usable candidate.
        p = np.array([.01, .02, .03, .04, .05, 0])
        order, events = residual_select(sparse.csr_matrix(x), p)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[-1], 1)
        self.assertNotIn(5, order)
        selected = [0]
        for i, _, _, score in events[1:]:
            design = np.column_stack([np.ones(len(x)), x[:, selected]])
            scores = {}
            for j in set(range(5)) - set(selected):
                residual = x[:, j] - design @ np.linalg.lstsq(design, x[:, j], rcond=None)[0]
                scores[j] = (np.corrcoef(x[:, j], residual)[0, 1]
                             if np.linalg.norm(residual) > 1e-10 else 0)
            self.assertAlmostEqual(score, max(scores.values()), places=7)
            self.assertAlmostEqual(score, scores[i], places=7)
            selected.append(i)
        p[1:5] = [.9, .8, .7, .6]
        self.assertEqual(residual_select(sparse.csr_matrix(x), p)[0], order)
        self.assertEqual(residual_select(sparse.csr_matrix((4, 0)), [])[0], [])
        self.assertEqual(residual_select(sparse.csr_matrix(np.ones((4, 2))), [0, 1])[0], [])
        self.assertEqual(residual_select(sparse.csr_matrix(x), p, maximum=1)[0], [0])

    def test_no_single_linkage_chaining(self):
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(rng.normal(size=(50, 3)))
        a = q[:, 0]; c = q[:, 1]
        x = np.column_stack([a, (a+c)/np.sqrt(2), c]) + 2
        _, _, groups = merge_features(sparse.csr_matrix(x), sparse.csr_matrix(x), np.ones(3), .6)
        self.assertGreater(len(groups), 1)

    def test_neighbors_exclude_self_and_handle_empty_or_constant(self):
        for x in (sparse.csr_matrix((4, 0)), sparse.csr_matrix(np.ones((4, 2))), sparse.csr_matrix(np.arange(4)[:, None])):
            result, _ = diagnostics(x)
            self.assertTrue(np.isnan(result["nearest_abs_median"]))
        result, _ = diagnostics(sparse.csr_matrix([[0., 3], [1, 2], [2, 1], [3, 0]]))
        self.assertAlmostEqual(result["nearest_abs_median"], 1)
        self.assertAlmostEqual(result["nearest_signed_median"], -1)
        self.assertEqual(result["high_correlation_pairs"], 1)

    def test_selection_does_not_depend_on_validation_values(self):
        x = sparse.csr_matrix([[0., 0, 0], [1, 2, 1], [2, 4, 0], [3, 6, 1]])
        table = pd.DataFrame(dict(effect=[.5,.49,.4], p=[.01,.02,.03], pair_count=[1,1,1]))
        for method, parameter in (("filter", .8), ("penalty", .5), ("merge", .8), ("residual", 0),
                                  ("penalty_mean", .5), ("penalty_r2", .5), ("penalty_residual_target", .5)):
            a = construct({"join": (x, x, table)}, method, parameter, target=[0,1,0,1])
            b = construct({"join": (x, x*0, table)}, method, parameter, target=[0,1,0,1])
            self.assertEqual(a[3], b[3])
            np.testing.assert_allclose(a[0].toarray(), b[0].toarray())

    def test_empty_classifier_and_effective_k(self):
        result, predictions = evaluate(sparse.csr_matrix((4,0)), sparse.csr_matrix((2,0)),
                                      np.array([0,1,0,1]), np.array([0,1]), [1,10])
        self.assertTrue((result.effective_k == 0).all())
        self.assertTrue((predictions.probability == .5).all())

    def test_updated_comparison_preserves_old_curves_and_excludes_unrequested_methods(self):
        from scripts.plot_updated_qinj_comparison import load_comparison, OLD_METHODS, NEW_METHODS
        def rows(methods, dataset="QinJ_2012", ks=range(1,11)):
            return pd.DataFrame([dict(method=m, requested_k=k, effective_k=k,
                source="real", dataset=dataset, auroc=.7, precision=.6, recall=.5, f1=.55)
                for m in methods for k in ks])
        original = rows([*OLD_METHODS, "jaccard_join"])
        side = rows([*NEW_METHODS, "baseline", "bacteria", "hybrid", "residual"])
        combined = load_comparison(original, side)
        self.assertEqual(set(combined.method), set(OLD_METHODS) | set(NEW_METHODS))
        self.assertEqual(len(combined), 100)
        np.testing.assert_array_equal(combined[combined.method.isin(OLD_METHODS)].auroc, .7)
        side.loc[side.method.eq("baseline"), "auroc"] = .8
        with self.assertRaises(ValueError):
            load_comparison(original, side)
        with self.assertRaises(ValueError):
            load_comparison(original.iloc[1:], rows([*NEW_METHODS, "baseline", "bacteria"]))
        coarse = (5,10,20,35,50,60,75,85,100)
        other = load_comparison(rows(OLD_METHODS, "Other", coarse),
                               rows([*NEW_METHODS, "baseline", "bacteria"], "Other", coarse),
                               "Other", coarse)
        self.assertEqual(set(other.dataset), {"Other"})
        self.assertEqual(len(other), 90)
        self.assertEqual(set(other.requested_k), set(coarse))

    def test_current_coverage_plot_preserves_protocol_locks_and_references(self):
        from scripts.plot_best_current_qinj_comparison import append_coverage
        ks = (1, 2)
        metrics = pd.DataFrame([dict(family=f, strength=.175 if f == "taxon_coverage" else 0,
            requested_k=k, effective_k=k, auroc=.7, precision=.6, recall=.5, f1=.55)
            for f in ("baseline", "bacteria", "taxon_coverage") for k in ks])
        original = metrics[metrics.family.ne("taxon_coverage")].copy()
        original["method"] = original.family.map({"baseline": "bernoulli_correlation_join",
                                                  "bacteria": "bh_bacteria"})
        locked = pd.DataFrame([dict(family="taxon_coverage", strength=.175, requested_k=2)])
        combined, label = append_coverage(original, metrics, locked, "refined", "refined grid", ks)
        self.assertEqual(len(combined), 6)
        self.assertIn("λ=0.175", label)
        self.assertTrue(combined.loc[combined.method.eq("refined"), "cv_selected_k"].eq(2).all())
        for bad in (metrics.iloc[1:], pd.concat([metrics, metrics.iloc[:1]]),
                    metrics.assign(strength=.25), metrics.assign(auroc=np.nan),
                    metrics.assign(auroc=.8)):
            with self.assertRaises(ValueError):
                append_coverage(original, bad, locked, "refined", "refined grid", ks)

        baseline_lock = pd.DataFrame([dict(family="baseline", strength=0, requested_k=1)])
        combined, label = append_coverage(original, metrics, baseline_lock, "refined", "refined grid", ks,
                                          dataset="Other", family="baseline")
        chosen = combined[combined.method.eq("refined")]
        self.assertTrue(chosen.family.eq("baseline").all())
        self.assertTrue(chosen.dataset.eq("Other").all())
        self.assertIn("CV retained", label)

    def test_coverage_aggregate_requires_complete_equal_dataset_weights(self):
        from scripts.plot_aggregate_coverage_comparison import aggregate, METHODS
        data = pd.DataFrame([dict(dataset=d, method=m, requested_k=k,
            auroc=score, precision=score, recall=score, f1=score)
            for d, score in (("A", .2), ("B", .8)) for m in METHODS for k in (1, 2)])
        mean, spread = aggregate(data, ["A", "B"], (1, 2))
        np.testing.assert_allclose(mean.auroc, .5)
        np.testing.assert_allclose(spread.auroc, np.std([.2, .8], ddof=1))
        self.assertTrue(mean.n_datasets.eq(2).all())
        for bad in (data.iloc[1:], pd.concat([data, data.iloc[:1]]), data.assign(auroc=np.nan)):
            with self.assertRaises(ValueError):
                aggregate(bad, ["A", "B"], (1, 2))


if __name__ == "__main__":
    unittest.main()
