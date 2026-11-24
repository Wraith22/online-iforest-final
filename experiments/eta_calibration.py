"""Pilot an unlabeled calibration rule for the Online iForest split threshold."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import file_digest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from depth_benchmark import DATASETS, ROOT, SEEDS, initialize_worker, load_features, run, run_once
from OnlineIForest.adaptive_eta_onlineiforest import AdaptiveEtaOnlineIForest


PILOT = ("Annthyroid", "Mammography", "Satellite", "NYC Taxi Shingle", "Shuttle", "SMTP", "INSECTS")
BASELINE_ETA = 32
CANDIDATES = (16, BASELINE_ETA, 64)


def proxy_anomalies(training, validation, rng):
    active = np.flatnonzero(np.ptp(training, axis=0) > 0)
    shuffled = validation.copy()
    for feature in active:
        shuffled[:, feature] = rng.permutation(validation[:, feature])

    tails = validation.copy()
    if active.size:
        extremes = np.quantile(training, [0.025, 0.975], axis=0)
        feature = rng.choice(active, size=len(tails))
        side = rng.integers(2, size=len(tails))
        tails[np.arange(len(tails)), feature] = extremes[side, feature]
    return shuffled, tails


def calibrate_eta(samples, batch_size, trees):
    training, validation = np.split(samples, [3 * len(samples) // 4])
    shuffled, tails = proxy_anomalies(training, validation, np.random.default_rng(0))
    evaluation = np.concatenate((validation, shuffled, tails))
    labels = np.repeat((0, 1), len(validation))
    proxy_auc = {}
    state = np.random.get_state()
    try:
        for eta in CANDIDATES:
            # Give each candidate the same random starting point and restore the live stream's RNG afterward.
            np.random.seed(0)
            model = AdaptiveEtaOnlineIForest(num_trees=trees, window_size=len(samples))
            model.split_eta = eta
            for batch in np.array_split(training, len(training) // batch_size + 1):
                model.learn_batch(batch)
            ordinary, shuffled_score, tail_score = np.split(model.score_batch(evaluation), 3)
            proxy_auc[eta] = tuple(
                roc_auc_score(labels, np.concatenate((ordinary, anomalous)))
                for anomalous in (shuffled_score, tail_score)
            )
    finally:
        np.random.set_state(state)

    baseline = proxy_auc[BASELINE_ETA]
    improvements = [
        eta for eta in CANDIDATES
        if all(current > original for current, original in zip(proxy_auc[eta], baseline))
    ]
    selected = max(improvements, key=lambda eta: sum(proxy_auc[eta]), default=BASELINE_ETA)
    return selected, proxy_auc


class CalibratedEtaForest(AdaptiveEtaOnlineIForest):
    """Select a threshold once using the first full window of observations."""

    def __init__(self, *, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.calibration = None

    def select_eta(self, samples):
        if self.calibration is None:
            self.calibration = calibrate_eta(samples, self.batch_size, self.num_trees)
        return self.calibration[0]


def compare(dataset, seed, digest, out):
    original = run(dataset, "code", seed, ROOT / "results/depth_benchmark", digest)
    checkpoint = out / f"{Path(DATASETS[dataset]).stem}_{seed}.json"
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        keys = ("dataset", "seed", "data_sha256")
        if all(result[key] == original[key] for key in keys) and result["fixed_auc"] == original["auc"]:
            return result
        raise ValueError(f"Results mismatch: {checkpoint}")

    stream = dataset == "INSECTS"
    X, y = load_features(ROOT / "data" / DATASETS[dataset], 10 if stream else None)
    args = SimpleNamespace(shuffle=not stream, batch=100, reset_at=[])
    model = CalibratedEtaForest(num_trees=32, batch_size=args.batch)
    with patch("run_experiment.make_model", return_value=model):
        auc, _, scores, _ = run_once(X, y, args, seed)
    if not np.isfinite(scores).all() or model.calibration is None:
        raise ValueError(f"Invalid calibration run for {dataset}, seed {seed}")
    selected, proxy_auc = model.calibration
    result = dict(dataset=dataset, seed=seed, data_sha256=digest, fixed_auc=original["auc"],
                  calibrated_auc=auc, selected_eta=selected)
    for eta, (shuffled, tails) in proxy_auc.items():
        result[f"shuffled_auc_eta_{eta}"] = shuffled
        result[f"tails_auc_eta_{eta}"] = tails
    temp = checkpoint.with_suffix(".tmp")
    temp.write_text(json.dumps(result) + "\n")
    temp.replace(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "results/eta_calibration")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    (ROOT / "results/depth_benchmark").mkdir(parents=True, exist_ok=True)
    datasets = tuple(DATASETS) if args.all_datasets else PILOT
    digests = {}
    for dataset in datasets:
        with (ROOT / "data" / DATASETS[dataset]).open("rb") as file:
            digests[dataset] = file_digest(file, "sha256").hexdigest()
    order = sorted(datasets, key=lambda dataset: (ROOT / "data" / DATASETS[dataset]).stat().st_size, reverse=True)
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(True,)) as pool:
        futures = [pool.submit(compare, dataset, seed, digests[dataset], args.out)
                   for dataset in order for seed in SEEDS]
        for future in as_completed(futures):
            row = future.result()
            runs.append(row)
            delta = row["calibrated_auc"] - row["fixed_auc"]
            print(f"{len(runs)}/{len(futures)} {row['dataset']} seed={row['seed']} "
                  f"eta={row['selected_eta']} delta={delta:+.6f}", flush=True)

    runs.sort(key=lambda row: (datasets.index(row["dataset"]), row["seed"]))
    with (args.out / "paired_auc.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(runs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)


if __name__ == "__main__":
    main()
