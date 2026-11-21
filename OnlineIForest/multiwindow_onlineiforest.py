"""An unsupervised ensemble of oIFOR models with different forgetting windows."""
from numbers import Integral

import numpy as np

from .onlineiforest import OnlineIForest
from .BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineIForest


class MultiWindowOnlineIForest(OnlineIForest):
    """Divide a fixed tree budget among windows and average normalized scores.

    Each forest is weighted by its tree count; any remainder goes to earlier windows.
    Each child normalizes by its own current sample count. Averaging raw depths
    with a shared normalizer would unfairly mix different window sizes. No label,
    drift location, or future sample is used to select a window or a weight.
    """

    def __init__(self, window_sizes=(2048, 4096), num_trees=32,
                 max_leaf_samples=32, type='adaptive', subsample=1.0,
                 branching_factor=2, metric='axisparallel', n_jobs=1,
                 depth_limit='code'):
        windows = tuple(window_sizes)
        if not windows or any(isinstance(w, bool) or not isinstance(w, Integral) or w <= 0 for w in windows):
            raise ValueError('window_sizes must contain positive integers')
        if len(set(windows)) != len(windows):
            raise ValueError('window_sizes must be distinct')
        if isinstance(num_trees, bool) or not isinstance(num_trees, Integral) or num_trees < len(windows):
            raise ValueError('num_trees must allocate at least one tree per window')
        super().__init__(num_trees, max(windows), branching_factor,
                         max_leaf_samples, type, subsample, n_jobs)
        self.window_sizes = windows
        quotient, remainder = divmod(num_trees, len(windows))
        self.tree_counts = tuple(quotient + (i < remainder) for i in range(len(windows)))
        self.weights = np.asarray(self.tree_counts, dtype=float) / num_trees
        self.models = [
            BoundedRandomProjectionOnlineIForest(
                num_trees=count, window_size=window, max_leaf_samples=max_leaf_samples,
                type=type, subsample=subsample, branching_factor=branching_factor,
                metric=metric, n_jobs=n_jobs, depth_limit=depth_limit,
            ) for window, count in zip(windows, self.tree_counts)
        ]
        self.trees = [tree for model in self.models for tree in model.trees]

    def learn_batch(self, data):
        for model in self.models:
            model.learn_batch(data)
        longest = max(self.models, key=lambda model: model.window_size)
        self.data_size = longest.data_size
        # Keep the base class's public window view consistent, without a third copy.
        self.data_window = longest.data_window
        return self

    def score_batch(self, data):
        """Tree-budget-weighted mean of the normalized forest scores."""
        return np.average([model.score_batch(data) for model in self.models],
                          axis=0, weights=self.weights)

    def predict_batch(self, data):
        """Weighted raw mean depth for diagnostics, not a shared score normalizer."""
        return np.average([model.predict_batch(data) for model in self.models],
                          axis=0, weights=self.weights)
