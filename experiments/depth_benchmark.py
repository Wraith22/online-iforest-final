"""Compare a depth formula, eta policy, or training window against the original program."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
from hashlib import file_digest
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run_experiment import load_dataset, run_once
from OnlineIForest.adaptive_eta_onlineiforest import AdaptiveEtaOnlineIForest
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
RESULT_DIRS = {None: "depth", "joint": "eta", "split-only": "eta_split", "trimmed": "eta_trimmed"}
FIELDS = ("dataset", "depth_limit", "seed", "auc", "rows", "features", "anomalies", "data_sha256")
PACKAGE = "OnlineIForest.BoundedRandomProjectionOnlineIForest"


def window_comparison_name(window):
    return "window" if window == 4096 else f"window_{window}"


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


def run(dataset, mode, seed, out, data_sha256, *, adaptive_eta=None, window=2048):
    path = ROOT / "data" / DATASETS[dataset]
    slug = Path(DATASETS[dataset]).stem
    eta_tag = {None: "", "joint": "auto_eta_", "split-only": "split_only_eta_", "trimmed": "trimmed_eta_"}[adaptive_eta]
    window_tag = f"w{window}_" if window != 2048 else ""
    name = f"{slug}_{mode}_{window_tag}{eta_tag}{seed}"
    checkpoint = out / f"{name}.json"
    expected = {"dataset": dataset, "depth_limit": mode, "seed": seed, "data_sha256": data_sha256}
    if adaptive_eta:
        expected["eta_policy"] = "adaptive"
    if window != 2048:
        expected["window"] = window
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if all(result.get(key) == value for key, value in expected.items()) and 0 <= result["auc"] <= 1:
            return result
        raise ValueError(f"Checkpoint differs from the current benchmark: {checkpoint}")

    stream = dataset == "INSECTS"
    X, y = load_features(path, 10 if stream else None)
    args = SimpleNamespace(
        method="oifor", windows=None, window=window, metric="axisparallel", trees=32,
        eta=32, depth_limit=mode, shuffle=not stream, batch=100, reset_at=[],
    )
    if adaptive_eta:
        model = AdaptiveEtaOnlineIForest(num_trees=args.trees, max_leaf_samples=args.eta, window_size=args.window,
                                         metric=args.metric, depth_limit=args.depth_limit,
                                         fixed_score_eta=adaptive_eta != "joint",
                                         trimmed_range=adaptive_eta == "trimmed")
        # Reuse the same shuffle, batches, and learn/score order as the fixed-eta run.
        with patch("run_experiment.make_model", return_value=model):
            auc, _, scores, labels = run_once(X, y, args, seed)
        eta_history = list(model.eta_counts.elements())
    else:
        auc, _, scores, labels = run_once(X, y, args, seed)
    if len(scores) != len(y) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid scores: {dataset}, {mode}, seed {seed}")
    result = dict(expected, auc=auc, rows=len(y), features=X.shape[1], anomalies=int(labels.sum()))
    if adaptive_eta:
        result.update(eta_min=min(eta_history), eta_max=max(eta_history), eta_median=float(np.median(eta_history)))
    temp = checkpoint.with_suffix(".tmp")
    temp.write_text(json.dumps(result) + "\n")
    temp.replace(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference-execution", action="store_true", help="Use the standard executor and NumPy calls")
    parser.add_argument("--out", type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--adaptive-eta", choices=["joint", "split-only", "trimmed"],
                            help="Compare an automatic eta policy with fixed eta 32")
    comparison.add_argument("--candidate-window", type=int,
                            help="Compare a training window against 2048 with eta 32 and the original depth formula")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.candidate_window is not None and (args.candidate_window < 1 or args.candidate_window == 2048):
        parser.error("--candidate-window must be positive and differ from 2048")
    depth_out = ROOT / "results/depth_benchmark"
    if args.candidate_window:
        comparison_name = window_comparison_name(args.candidate_window)
    else:
        comparison_name = RESULT_DIRS[args.adaptive_eta]
    out = args.out or ROOT / f"results/{comparison_name}_benchmark"
    if (args.adaptive_eta or args.candidate_window) and out.resolve() == depth_out.resolve():
        parser.error("Use a separate output directory for the comparison")
    out.mkdir(parents=True, exist_ok=True)

    digests = {}
    for dataset, name in DATASETS.items():
        with (ROOT / "data" / name).open("rb") as file:
            digests[dataset] = file_digest(file, "sha256").hexdigest()

    # Start long data streams first to keep workers busy until the final short runs.
    order = sorted(DATASETS, key=lambda name: (ROOT / "data" / DATASETS[name]).stat().st_size, reverse=True)
    if args.adaptive_eta or args.candidate_window:
        depth_out.mkdir(parents=True, exist_ok=True)
        configs = [("code", None, 2048, depth_out), ("code", args.adaptive_eta, args.candidate_window or 2048, out)]
    else:
        configs = [(mode, None, 2048, out) for mode in MODES]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(not args.reference_execution,)) as pool:
        futures = [pool.submit(run, dataset, mode, seed, directory, digests[dataset], adaptive_eta=adaptive_eta, window=window)
                   for dataset in order for seed in SEEDS for mode, adaptive_eta, window, directory in configs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            eta_policy = row.get("eta_policy", "fixed")
            print(f"{len(results):3}/{len(futures)} {row['dataset']:17} {row['depth_limit']:5} "
                  f"eta={eta_policy:8} window={row.get('window', 2048)} seed={row['seed']} AUC={row['auc']:.8f}", flush=True)

    results.sort(key=lambda row: (list(DATASETS).index(row["dataset"]), row["seed"], MODES.index(row["depth_limit"]),
                                  row.get("eta_policy") == "adaptive", row.get("window", 2048) != 2048))
    fields = FIELDS
    if args.adaptive_eta:
        fields = (*FIELDS, "eta_policy", "score_eta", "eta_span", "eta_min", "eta_max", "eta_median")
        for row in results:
            if "eta_policy" not in row:
                row.update(eta_policy="fixed", score_eta="fixed", eta_span="none", eta_min=32, eta_max=32, eta_median=32)
            else:
                row["score_eta"] = "adaptive" if args.adaptive_eta == "joint" else "fixed"
                row["eta_span"] = "trimmed" if args.adaptive_eta == "trimmed" else "full"
    if args.candidate_window:
        fields = (*FIELDS, "window")
        for row in results:
            row.setdefault("window", 2048)
    with (out / "paired_auc.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
