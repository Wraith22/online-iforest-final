"""Check exact agreement between the Python and compiled benchmarks."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from depth_benchmark import DATASETS, ROOT, initialize_worker, load_features, run_once
from native_benchmark import compile_benchmark, run_native


def check_dataset(dataset, library):
    stream = dataset == "INSECTS"
    X, y = load_features(ROOT / "data" / DATASETS[dataset], 10 if stream else None)
    size = 5600  # Exercise forgetting even with the larger 4096-row window.
    subset = np.arange(size) if stream else np.random.RandomState(41).permutation(len(y))[:size]
    if np.unique(y[subset]).size == 1:
        subset[-1] = np.flatnonzero(y != y[subset[0]])[0]
    X, y = X[subset], y[subset]

    seed = 11
    for mode, window in (("code", 2048), ("paper", 2048), ("code", 4096)):
        args = SimpleNamespace(method="oifor", windows=None, window=window, metric="axisparallel",
                               trees=32, eta=32, depth_limit=mode, shuffle=not stream, batch=100, reset_at=[])
        auc, _, scores, labels = run_once(X, y, args, seed)
        native_auc, native_scores, native_labels = run_native(X, y, window, mode, args.shuffle, seed, library)
        if not np.array_equal(scores, native_scores) or not np.array_equal(labels, native_labels) or auc != native_auc:
            raise AssertionError(f"Benchmark mismatch for {dataset}, {mode}, window {window}")
    return dataset


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        library = compile_benchmark(Path(directory))
        with ProcessPoolExecutor(max_workers=4, initializer=initialize_worker, initargs=(True,)) as pool:
            for dataset in pool.map(check_dataset, DATASETS, [library] * len(DATASETS)):
                print(f"{dataset}: identical scores, labels, and AUC for all three settings", flush=True)
