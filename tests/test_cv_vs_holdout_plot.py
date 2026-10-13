"""Cache-only comparison must keep evaluation types and missing results explicit."""
import unittest
import matplotlib.pyplot as plt
import pandas as pd
from scripts import plot_cv_vs_holdout as plot


class ComparisonPlotTests(unittest.TestCase):
    def setUp(self):
        self.completed = plot.DATASETS[:-1]
        ks = sorted(set(plot.GRIDS['fine']) | set(plot.GRIDS['coarse']))
        self.holdout = pd.DataFrame([
            dict(dataset=d, method=m, requested_k=k, auroc=.8)
            for d in plot.DATASETS for m in plot.LABELS for k in ks])
        self.cv = self.holdout[self.holdout.dataset.isin(self.completed)].copy()
        self.cv['auroc'], self.cv['n_folds'] = .7, 5

    def test_same_colors_solid_cv_dotted_holdout_both_grids(self):
        table = plot.comparison_table(self.cv, self.holdout, self.completed)
        for grid in plot.GRIDS:
            figure, axes = plot.plot_comparison(table, grid)
            try:
                self.assertEqual(len(axes.flat[0].lines), 6)
                for index, method in enumerate(plot.LABELS):
                    cv, holdout = axes.flat[0].lines[index * 2:index * 2 + 2]
                    self.assertEqual(cv.get_color(), plot.COLORS[method])
                    self.assertEqual(cv.get_color(), holdout.get_color())
                    self.assertEqual((cv.get_linestyle(), holdout.get_linestyle()), ('-', ':'))
                    self.assertTrue((cv.get_ydata() == .7).all())
                    self.assertTrue((holdout.get_ydata() == .8).all())
                    self.assertEqual(list(cv.get_xdata()), plot.GRIDS[grid])
                missing = axes.flat[-1]
                self.assertEqual(len(missing.lines), 3)
                self.assertTrue(all(line.get_linestyle() == ':' for line in missing.lines))
                self.assertIn('CV unavailable', missing.texts[0].get_text())
                self.assertEqual(len(figure.legends[0].get_texts()), 6)
            finally:
                plt.close(figure)

    def test_rejects_duplicates_missing_points_invalid_auc_and_partial_folds(self):
        bad_tables = [pd.concat([self.cv, self.cv.iloc[:1]]), self.cv.iloc[1:],
                      self.cv.assign(auroc=float('nan')), self.cv.assign(auroc=1.1),
                      self.cv.assign(n_folds=4)]
        for invalid in bad_tables:
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                plot.comparison_table(invalid, self.holdout, self.completed)
        with self.assertRaises(ValueError):
            plot.comparison_table(self.cv, self.holdout, plot.DATASETS)

    def test_tight_shared_limits_include_fine_only_extreme_in_both_figures(self):
        self.holdout.loc[self.holdout.requested_k.eq(9), 'auroc'] = .28571429
        self.holdout.loc[self.holdout.requested_k.eq(100), 'auroc'] = 1.0
        table = plot.comparison_table(self.cv, self.holdout, self.completed)
        for grid in plot.GRIDS:
            figure, axes = plot.plot_comparison(table, grid)
            try:
                for axis in axes.flat:
                    self.assertAlmostEqual(axis.get_ylim()[0], .27)
                    self.assertAlmostEqual(axis.get_ylim()[1], 1.01)
                    self.assertLess(axis.get_ylim()[0], table.auroc.min())
                    self.assertGreater(axis.get_ylim()[1], table.auroc.max())
            finally:
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
