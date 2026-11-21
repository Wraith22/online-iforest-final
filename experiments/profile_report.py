"""Profile one independent oIFOR run for the experiment report (Linux)."""

import argparse
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from run_experiment import load_dataset, run_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--depth-limit", choices=("code", "paper"), default="code")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--output", type=Path, required=True, help="Output path prefix")
    args = parser.parse_args()
    args.method = "oifor"
    args.metric = "axisparallel"
    args.trees = args.eta = 32
    args.batch = 100
    args.reset_at = []

    start = time.perf_counter()
    features, labels = load_dataset(args.dataset, args.max_features)
    auc, elapsed, scores, ordered_labels = run_once(features, labels, args, args.seed)
    pipeline_s = time.perf_counter() - start
    # Linux ru_maxrss counts KiB, including interpreter, loaded data and libraries.
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "dataset": str(args.dataset),
        "window": args.window,
        "depth_limit": args.depth_limit,
        "seed": args.seed,
        "n": len(labels),
        "trees": args.trees,
        "eta": args.eta,
        "metric": args.metric,
        "batch": args.batch,
        "shuffle": args.shuffle,
        "max_features": args.max_features,
        "auc": auc,
        "learn_score_s": elapsed,
        "load_through_auc_s": pipeline_s,
        "amortized_us_per_row": 1e6 * elapsed / len(labels),
        "peak_rss_kib": peak_rss_kib,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "system": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), scores=scores, labels=ordered_labels)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
