"""Online Isolation Forest with a label-free split threshold."""

from collections import Counter

import numpy as np

from .BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineIForest


class AdaptiveEtaOnlineIForest(BoundedRandomProjectionOnlineIForest):
    """Re-estimate eta after each window's worth of new observations."""

    def __init__(self, *, window_size: int = 2048, max_leaf_samples: int = 32,
                 fixed_score_eta: bool = True, trimmed_range: bool = False, **kwargs):
        if window_size < 1 or max_leaf_samples < 1:
            raise ValueError("The window and initial eta must be positive")
        super().__init__(window_size=window_size, max_leaf_samples=max_leaf_samples, **kwargs)
        self.initial_eta = max_leaf_samples
        self.split_eta = max_leaf_samples
        self.fixed_score_eta = fixed_score_eta
        self.trimmed_range = trimmed_range
        self.eta_counts = Counter({max_leaf_samples: 1})
        self._samples_since_selection = 0

    def learn_batch(self, data: np.ndarray) -> "AdaptiveEtaOnlineIForest":
        if len(data) and self._samples_since_selection >= self.window_size:
            self.split_eta = self.select_eta(np.asarray(self.data_window))
            if not self.fixed_score_eta:
                self.max_leaf_samples = self.split_eta
            self.eta_counts[self.split_eta] += 1
            self._samples_since_selection = 0

        for tree in self.trees:
            tree.max_leaf_samples = self.split_eta
        try:
            super().learn_batch(data)
        finally:
            # Keep score normalization and leaf depth correction in the original units.
            if self.fixed_score_eta and self.split_eta != self.initial_eta:
                for tree in self.trees:
                    tree.max_leaf_samples = self.initial_eta
        self._samples_since_selection += len(data)
        return self

    def select_eta(self, samples: np.ndarray) -> int:
        """Keep eta near its initial value for roughly uniform features."""
        if self.trimmed_range:
            low, q1, q3, high = np.quantile(samples, [0.05, 0.25, 0.75, 0.95], axis=0)
            span = high - low
            central_span = q3 - q1
            uniform_fraction = 0.5 / 0.9
        else:
            span = np.ptp(samples, axis=0)
            q1, q3 = np.quantile(samples, [0.25, 0.75], axis=0)
            central_span = q3 - q1
            uniform_fraction = 0.5
        active = span > 0
        if not np.any(active):
            return self.initial_eta

        central_fraction = np.median(central_span[active] / span[active])
        # Use a smaller split threshold when most points occupy a narrow range.
        # The lower bound keeps the tree from splitting on very few points.
        eta = round(self.initial_eta * np.sqrt(central_fraction / uniform_fraction))
        return max(1, self.initial_eta // 4, eta)
