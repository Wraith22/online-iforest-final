import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from OnlineIForest import OnlineIForest


class MultiWindowTests(unittest.TestCase):
    def setUp(self):
        self.data = np.random.default_rng(12).normal(size=(180, 3))

    def model(self, **kwargs):
        return OnlineIForest.create('multiwindowonlineiforest', **kwargs)

    def test_one_window_matches_original_scores_and_random_state(self):
        results = []
        states = []
        for kind, window in [('boundedrandomprojectiononlineiforest', {'window_size': 64}),
                             ('multiwindowonlineiforest', {'window_sizes': [64]})]:
            np.random.seed(7)
            model = OnlineIForest.create(kind, num_trees=5, max_leaf_samples=4, **window)
            results.append([model.learn_batch(batch).score_batch(batch)
                            for batch in np.array_split(self.data, 9)])
            states.append(np.random.random(10))
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(states[0], states[1])

    def test_budget_weights_and_distinct_forgetting_windows(self):
        model = self.model(window_sizes=[32, 64], num_trees=5, max_leaf_samples=4)
        self.assertEqual(model.tree_counts, (3, 2))
        self.assertEqual(len(model.trees), 5)
        for batch in np.array_split(self.data, 9):
            model.learn_batch(batch)
        for child, window in zip(model.models, [32, 64]):
            self.assertEqual(child.data_size, window)
            self.assertEqual(len(child.data_window), window)
            np.testing.assert_array_equal(child.data_window, self.data[-window:])
            self.assertEqual(child.trees[0].root.data_size, window)
        self.assertEqual(model.data_size, 64)
        self.assertIs(model.data_window, model.models[1].data_window)
        expected = .6 * model.models[0].score_batch(self.data) + .4 * model.models[1].score_batch(self.data)
        np.testing.assert_allclose(model.score_batch(self.data), expected)
        scores = model.score_batch(self.data)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_batch_larger_than_every_window(self):
        model = self.model(window_sizes=[32, 64], num_trees=4, max_leaf_samples=4)
        model.learn_batch(self.data)
        for child, window in zip(model.models, [32, 64]):
            np.testing.assert_array_equal(child.data_window, self.data[-window:])
        self.assertTrue(np.isfinite(model.predict_batch(self.data)).all())

    def test_invalid_configuration(self):
        for windows in ([], [0, 32], [-1], [32, 32], [32.5], [True]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                self.model(window_sizes=windows)
        with self.assertRaises(ValueError):
            self.model(window_sizes=[32, 64], num_trees=1)

    def test_cli_outputs_and_exclusive_windows(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            labels = np.arange(len(self.data)) % 10 == 0
            data = np.column_stack([np.arange(len(self.data)), self.data, labels])
            np.savetxt(path/'sample.csv', data, delimiter=',', header='id,x,y,z,label', comments='')
            command = [sys.executable, 'experiments/run_experiment.py', '--dataset', str(path/'sample.csv'),
                       '--windows', '32', '64', '--trees', '4', '--eta', '4', '--runs', '1', '--out', directory]
            subprocess.run(command, cwd=root, check=True, capture_output=True)
            prefix = path/'sample_oifor_mw32-64'
            config = json.loads(Path(str(prefix)+'_config.json').read_text())
            self.assertEqual(config['windows'], [32, 64])
            self.assertEqual(config['trees'], 4)
            self.assertEqual(config['evaluation_order'], 'learn_then_score')
            with np.load(str(prefix)+'_run0.npz') as output:
                self.assertEqual(output['scores'].shape, (180,))
                self.assertEqual(output['labels'].sum(), labels.sum())
            invalid = subprocess.run(command + ['--window', '16'], cwd=root, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
