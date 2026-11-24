"""Run the fixed-eta binary benchmark with its compiled implementation."""

from ctypes import CDLL, c_int
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from numpy.ctypeslib import ndpointer
from sklearn.metrics import roc_auc_score

from OnlineIForest.onlineitree import OnlineITree


def compile_benchmark(directory):
    source = Path(__file__).with_name("binary_forest_benchmark.cc")
    library = directory / "binary_forest_benchmark.so"
    with TemporaryDirectory(dir=directory) as temporary:
        compiled = Path(temporary) / library.name
        subprocess.run(["g++", "-std=c++17", "-O3", "-ffp-contract=off", "-fPIC", "-shared",
                        str(source), "-o", str(compiled)], check=True)
        compiled.replace(library)
    return library.resolve()


@lru_cache(maxsize=1)
def load_benchmark(library):
    run = CDLL(str(library)).benchmark_binary_forest
    doubles = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    state = ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")
    run.argtypes = (doubles, c_int, c_int, c_int, c_int, state, c_int, doubles, doubles, c_int, doubles)
    run.restype = c_int
    return run


@lru_cache(maxsize=3)
def path_lengths(length, branching):
    return np.fromiter(
        (OnlineITree.get_random_path_length(branching, 32, n) for n in range(length)),
        dtype=float, count=length,
    )


def run_native(X, y, window, mode, shuffle, seed, library):
    rng = np.random.RandomState(seed)
    if shuffle:
        order = rng.permutation(len(X))
        X, y = X[order], y[order]
    X = np.ascontiguousarray(X, dtype=float)
    _, state, position, _, _ = rng.get_state()
    n, features = X.shape
    limits = path_lengths(min(n, window + 100) + 1, 1 if mode == "paper" else 2)
    corrections = path_lengths(n + 1, 2)
    depths = np.empty((32, n), dtype=float)
    status = load_benchmark(library)(X, n, features, window, int(np.ceil(limits[-1])),
                                     state, position, limits, corrections, len(corrections), depths)
    if status:
        raise RuntimeError("Compiled benchmark failed")

    scores = np.empty(n, dtype=float)
    eps = np.finfo(float).eps
    # Keep the same reduction shape and score normalization as the reference.
    for batch in np.array_split(np.arange(n), n // 100 + 1):
        start, stop = batch[0], batch[-1] + 1
        average_depth = np.ascontiguousarray(depths[:, start:stop]).T.mean(axis=1)
        scores[start:stop] = 2 ** (-average_depth / (corrections[min(stop, window)] + eps))
    return roc_auc_score(y, scores), scores, y
