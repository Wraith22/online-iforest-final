"""Compare the original depth limit and the paper depth limit on the 12 datasets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
from hashlib import file_digest
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_experiment import load_dataset, run_once
from OnlineIForest.onlineitree import OnlineITree


ROOT = Path(__file__).resolve().parents[1]
INSECTS_FILE = "insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv"
DATASETS = {
    "Donors": "stationary/hst/donors.csv",
    "HTTP": "stationary/ifor/Http.csv",
    "Forest Cover": "stationary/ifor/ForestCover.csv",
    "Fraud": "stationary/hst/fraud.csv",
    "Mulcross": "stationary/ifor/Mulcross.csv",
    "SMTP": "stationary/ifor/Smtp.csv",
    "Shuttle": "stationary/ifor/Shuttle.csv",
    "Mammography": "stationary/ifor/Mammography.csv",
    "NYC Taxi Shingle": "stationary/rrcf/nyc_taxi_shingle.csv",
    "Annthyroid": "stationary/ifor/Annthyroid.csv",
    "Satellite": "stationary/ifor/Satellite.csv",
    "INSECTS": f"non-stationary/kqt/{INSECTS_FILE}",
}
SEEDS = range(5)
MODES = ("code", "paper")
FIELDS = ("dataset", "depth_limit", "seed", "auc", "rows", "features", "anomalies", "data_sha256")
PACKAGE = "OnlineIForest.BoundedRandomProjectionOnlineIForest"


class InlineExecutor:
    """Run the forest's one-worker map in the same order without creating threads."""

    def __init__(self, max_workers):
        if max_workers != 1:
            raise ValueError("Inline benchmark execution requires one worker per forest")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    map = staticmethod(map)


def split_binary(self, data, projection_vector, split_values):
    projected = data @ projection_vector
    indices = np.argsort(projected)
    boundary = projected[indices].searchsorted(split_values[0])
    return [indices[:boundary], indices[boundary:]]


def stack_arrays(arrays):
    return np.concatenate([array[None, :] if array.ndim == 1 else array for array in arrays])


def initialize_worker(use_inline):
    if not use_inline:
        return
    # These operations are equivalent for the benchmark's binary forest.
    forest = import_module(f"{PACKAGE}.boundedrandomprojection_onlineiforest")
    tree = import_module(f"{PACKAGE}.boundedrandomprojection_onlineitree")
    forest.ThreadPoolExecutor = InlineExecutor
    tree.BoundedRandomProjectionOnlineITree.split_data = split_binary
    tree.vstack = stack_arrays
    OnlineITree.get_random_path_length = staticmethod(lru_cache(maxsize=8192)(OnlineITree.get_random_path_length))


@lru_cache(maxsize=2)
def load_features(path, features):
    X, y = load_dataset(path, features)
    if not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected finite features and two binary labels in {path}")
    return X, y


def run(dataset, mode, seed, out, data_sha256):
    path = ROOT / "data" / DATASETS[dataset]
    slug = Path(DATASETS[dataset]).stem
    checkpoint = out / f"{slug}_{mode}_{seed}.json"
    expected = {"dataset": dataset, "depth_limit": mode, "seed": seed, "data_sha256": data_sha256}
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if all(result.get(key) == value for key, value in expected.items()) and 0 <= result["auc"] <= 1:
            return result
        raise ValueError(f"Checkpoint differs from the current benchmark: {checkpoint}")

    stream = dataset == "INSECTS"
    X, y = load_features(path, 10 if stream else None)
    args = SimpleNamespace(
        method="oifor", windows=None, window=2048, metric="axisparallel", trees=32,
        eta=32, depth_limit=mode, shuffle=not stream, batch=100, reset_at=[],
    )
    auc, _, scores, labels = run_once(X, y, args, seed)
    if len(scores) != len(y) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid scores: {dataset}, {mode}, seed {seed}")
    result = dict(expected, auc=auc, rows=len(y), features=X.shape[1], anomalies=int(labels.sum()))
    temp = checkpoint.with_suffix(".tmp")
    temp.write_text(json.dumps(result) + "\n")
    temp.replace(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference-execution", action="store_true", help="Use the standard executor and NumPy calls")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "depth_benchmark")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    digests = {}
    for dataset, name in DATASETS.items():
        with (ROOT / "data" / name).open("rb") as file:
            digests[dataset] = file_digest(file, "sha256").hexdigest()

    # Start long data streams first to keep workers busy until the final short runs.
    order = sorted(DATASETS, key=lambda name: (ROOT / "data" / DATASETS[name]).stat().st_size, reverse=True)
    jobs = [(dataset, mode, seed, args.out, digests[dataset]) for dataset in order for seed in SEEDS for mode in MODES]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(not args.reference_execution,)) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{len(results):3}/{len(jobs)} {row['dataset']:17} {row['depth_limit']:5} seed={row['seed']} AUC={row['auc']:.8f}", flush=True)

    results.sort(key=lambda row: (list(DATASETS).index(row["dataset"]), row["seed"], MODES.index(row["depth_limit"])))
    with (args.out / "paired_auc.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
