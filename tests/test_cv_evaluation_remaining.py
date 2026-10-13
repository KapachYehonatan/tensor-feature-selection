"""Extension scope and combined views must preserve the completed pilot."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from scripts import run_cv_evaluation_remaining as extension


class ExtensionTests(unittest.TestCase):
    def test_six_new_cohorts_keep_all_existing_science_settings(self):
        self.assertEqual(len(extension.DATASETS), 6)
        self.assertFalse(set(extension.DATASETS) & set(extension.PILOT_DATASETS))
        self.assertEqual(set(extension.DATASETS), set(extension.THRESHOLDS))
        self.assertEqual(extension.study.FOLDS, 5)
        self.assertEqual(extension.study.SEED, 42)
        self.assertEqual(extension.study.EDGE_LIMIT, 500_000_000)
        self.assertEqual(extension.study.MEMORY_LIMIT_GIB, 200)
        self.assertEqual((extension.study.JOIN_WORKERS, extension.study.COMMUNITY_WORKERS), (4, 1))

    def test_pilot_compatibility_refuses_incomplete_or_changed_cv(self):
        s = extension.study
        protocol = dict(signature='pilot', folds=s.FOLDS, seed=s.SEED, k_values=s.K_VALUES,
            tune_k=False, permanent_holdout=False, methods=s.METHODS, bh_fdr=.10, prevalence=[.05, 5],
            memory_limit_gib=s.MEMORY_LIMIT_GIB, thresholds={d: extension.ALL_THRESHOLDS[d] for d in extension.PILOT_DATASETS},
            model='Training-scaled balanced L2 C=1', leiden=dict(resolution=1., seed=17729, iterations=-1, objective='modularity', weight='abs(phi)'))
        result = dict(status='complete', source_caches_unchanged=True, signature='pilot', completed_datasets=extension.PILOT_DATASETS)
        with tempfile.TemporaryDirectory() as name, patch.object(extension, 'PILOT', Path(name)):
            root = Path(name)
            (root/'protocol.json').write_text(json.dumps(protocol))
            (root/'result.json').write_text(json.dumps(result))
            extension.verify_pilot()
            protocol['seed'] = 43
            (root/'protocol.json').write_text(json.dumps(protocol))
            with self.assertRaises(ValueError): extension.verify_pilot()

    def test_combined_partial_and_complete_grids_do_not_modify_pilot(self):
        s = extension.study
        original = s.OUTPUT, s.DATASETS, s.THRESHOLDS
        with tempfile.TemporaryDirectory() as name:
            pilot = Path(name)/'pilot'
            pilot.mkdir()
            remaining, combined = pilot/'remaining', pilot/'all_real'
            remaining.mkdir()
            (pilot/'protocol.json').write_text(json.dumps(dict(signature='pilot')))
            (remaining/'protocol.json').write_text(json.dumps(dict(signature='remaining')))
            for dataset in extension.PILOT_DATASETS:
                (pilot/dataset).mkdir()
                (pilot/dataset/'unchanged.txt').write_text('Completed scientific results')
            for dataset in extension.DATASETS: (remaining/dataset).mkdir()
            pd.DataFrame([dict(dataset=d, fold=f, status='complete') for d in extension.PILOT_DATASETS for f in range(5)]).to_csv(pilot/'status.csv', index=False)
            pd.DataFrame([dict(dataset=d, fold=f, status='complete') for d in extension.DATASETS for f in range(5)]).to_csv(remaining/'status.csv', index=False)
            def report(complete):
                records = [dict(dataset=d, method=m, requested_k=k, auroc=.75, precision=.7, recall=.8, f1=.746)
                           for d in complete for m in s.METHODS for k in s.K_VALUES]
                pd.DataFrame(records).to_csv(s.OUTPUT/'mean_fold_metrics.csv', index=False)
                (s.OUTPUT/'REPORT.md').write_text('# QinJ/HMP: five-fold full-pipeline evaluation\nPredeclared absolute-phi thresholds: QinJ .80, HMP .70;')
            with (patch.object(extension, 'PILOT', pilot), patch.object(extension, 'OUTPUT', remaining),
                  patch.object(extension, 'ALL_OUTPUT', combined), patch.object(extension, 'BASE_REPORT', side_effect=report)):
                protected = extension.protected_states()
                extension.consolidate([])
                self.assertEqual(json.loads((combined/'result.json').read_text())['status'], 'partial')
                self.assertEqual(len(pd.read_csv(combined/'mean_fold_metrics.csv')), 2*3*17)
                (remaining/'result.json').write_text(json.dumps(dict(status='complete', source_caches_unchanged=True)))
                extension.consolidate(extension.DATASETS)
                marker = json.loads((combined/'result.json').read_text())
                self.assertEqual(marker['status'], 'complete')
                self.assertFalse(marker['pilot_refitted'])
                self.assertEqual(len(pd.read_csv(combined/'mean_fold_metrics.csv')), 8*3*17)
                self.assertEqual(len(pd.read_csv(combined/'status.csv')), 40)
                self.assertEqual(protected, extension.protected_states())
                for dataset in [*extension.PILOT_DATASETS, *extension.DATASETS]:
                    self.assertTrue((combined/dataset).is_symlink())
                    root = pilot if dataset in extension.PILOT_DATASETS else remaining
                    self.assertEqual((combined/dataset).resolve(), (root/dataset).resolve())
                self.assertIn('ZhuF_2020 0.75', (combined/'REPORT.md').read_text())
                self.assertEqual(original, (s.OUTPUT, s.DATASETS, s.THRESHOLDS))
                with patch.object(extension, 'BASE_REPORT', side_effect=RuntimeError('Report failure')):
                    with self.assertRaises(RuntimeError): extension.consolidate(extension.DATASETS)
                self.assertEqual(original, (s.OUTPUT, s.DATASETS, s.THRESHOLDS))


if __name__ == '__main__': unittest.main()
