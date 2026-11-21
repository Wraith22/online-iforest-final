"""Verify oIFOR run artifacts and render the experiment report and its figures."""

import argparse
from dataclasses import dataclass, field
import csv
import hashlib
import json
from math import log
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from windowed_auc import windowed_auc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "experiment_report"
ARCHIVE = RESULTS / "multiwindow_review"
ANN_TUNING = RESULTS / "window_only_ann"
INSECTS_TUNING = RESULTS / "window_only_insects"
ATTACHMENTS = ROOT.parent / "attachments"
ANN = "Annthyroid"
INSECTS = "insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male"
ANN_PATH = f"data/stationary/ifor/{ANN}.csv"
INSECTS_PATH = f"data/non-stationary/kqt/{INSECTS}.csv"
FIGURES = ROOT / "figures"
DOC = ROOT / "docs"
WIDTH = 5000
STEP = 1000
# Known drift markers used by the original INSECTS reset experiment.
DRIFT_POINTS = (51400, 74500, 101000, 142500, 152000)
ATOL = 1e-13


@dataclass
class Group:
    summary: pd.DataFrame
    records: dict
    curves: dict = field(default_factory=dict)

    @property
    def auc(self):
        return self.summary.set_index("seed").auc


def prefix(directory, dataset, tag=""):
    suffix = "_" + tag if tag else ""
    return directory / f"{dataset}_oifor{suffix}"


def cumulative_auc(scores, labels, stops):
    # The full-stream score ranks give all prefixes the same sorted bins.
    # Empty bins in a prefix do not affect its positive-negative pair count.
    _, rank = np.unique(scores, return_inverse=True)
    bins = int(rank.max()) + 1
    positive = labels == 1
    values = []
    for stop in stops:
        pos_counts = np.bincount(rank[:stop][positive[:stop]], minlength=bins)
        neg_counts = np.bincount(rank[:stop][~positive[:stop]], minlength=bins)
        pairs = pos_counts.sum() * neg_counts.sum()
        concordant = np.dot(pos_counts, np.cumsum(neg_counts) - 0.5 * neg_counts)
        values.append(concordant / pairs if pairs else np.nan)
    for i in (0, len(stops)//2, len(stops)-1):
        if np.isfinite(values[i]):
            stop = stops[i]
            np.testing.assert_allclose(values[i], roc_auc_score(labels[:stop], scores[:stop]),
                                       atol=ATOL, rtol=0)
    return values


def inspect_record(path, expected_labels, expected_auc, shuffled):
    with np.load(path, allow_pickle=False) as raw:
        scores, labels = raw["scores"], raw["labels"]
    assert scores.shape == labels.shape == expected_labels.shape, path
    assert np.isfinite(scores).all() and np.isin(labels, [0, 1]).all(), path
    if shuffled:
        assert labels.sum() == expected_labels.sum(), path
    else:
        np.testing.assert_array_equal(labels, expected_labels, err_msg=str(path))
    np.testing.assert_allclose(roc_auc_score(labels, scores), expected_auc, rtol=0, atol=ATOL, err_msg=str(path))
    return scores, labels


def group(paths, seeds, labels, *, scores_dir=None, uploaded_ann=False, shuffled=False,
          expected_window=None, expected_depth="code"):
    rows = []
    records = {}
    for path in paths:
        summary = pd.read_csv(path.with_name(path.name + "_summary.csv"))
        assert not summary.empty and summary.seed.is_unique, path
        if expected_window is not None:
            config = json.loads(path.with_name(path.name + "_config.json").read_text())
            assert config["runs"] == len(summary) and config["seed"] == summary.seed.iloc[0], path
            assert config["evaluation_order"] == "learn_then_score" and config["shuffle"] == shuffled, path
            assert config["trees"] == config["eta"] == 32 and config["batch"] == 100, path
            assert config["metric"] == "axisparallel" and config["reset_at"] == [], path
            if isinstance(expected_window, tuple):
                assert config["windows"] == list(expected_window), path
            else:
                assert config["window"] == expected_window and not config.get("windows"), path
            assert config["depth_limit"] == expected_depth, path
            assert config["max_features"] == (None if shuffled else 10), path
        for row in summary.itertuples(index=False):
            filename = f"{path.name}_run{row.run}.npz"
            if uploaded_ann:
                candidates = sorted(ATTACHMENTS.glob(f"*/{filename}"))
                if not candidates:
                    raise FileNotFoundError(f"Missing attached score file: {filename}")
                score_path = candidates[0]
            else:
                score_path = (scores_dir or path.parent) / filename
            records[row.seed] = inspect_record(score_path, labels, row.auc, shuffled)
            if shuffled:
                order = np.random.RandomState(row.seed).permutation(len(labels))
                np.testing.assert_array_equal(records[row.seed][1], labels[order], err_msg=str(score_path))
            rows.append({"seed": row.seed, "auc": row.auc, "time_s": row.time_s})
    frame = pd.DataFrame(rows).sort_values("seed")
    assert frame.seed.tolist() == list(seeds), paths
    return Group(frame, records)


def summarize(values):
    q1, q3 = values.quantile([0.25, 0.75])
    return f"| {len(values)} | {values.median():.6f} | {values.mean():.6f} ± {values.std(ddof=1):.6f} | {q1:.6f}–{q3:.6f} | {q3-q1:.6f} |"


def paired(before, after):
    pd.testing.assert_index_equal(before.auc.index, after.auc.index)
    delta = after.auc - before.auc
    wins, losses = (delta > 0).sum(), (delta < 0).sum()
    ties = len(delta) - wins - losses
    p = wilcoxon(delta, alternative="two-sided", method="exact").pvalue if not ties else (
        wilcoxon(delta, alternative="two-sided", method="auto").pvalue if wins + losses else 1.0
    )
    return int(wins), int(losses), int(ties), float(p), delta


def contrast(before, after):
    delta = after.auc.median() - before.auc.median()
    return f"{delta:+.6f} / {100*delta/before.auc.median():+.2f}%"


def matrix(data, cumulative=False):
    curves = []
    for scores, labels in data.records.values():
        if cumulative:
            x = np.r_[np.arange(WIDTH, len(labels), STEP), len(labels)]
            y = cumulative_auc(scores, labels, x)
        else:
            x, y = windowed_auc(scores, labels, WIDTH, STEP)
        curves.append(y)
    return x, np.asarray(curves)


def median_curve(data, cumulative=False):
    if cumulative in data.curves:
        return data.curves[cumulative]
    x, ys = matrix(data, cumulative)
    assert not np.isnan(ys).all(axis=0).any()
    data.curves[cumulative] = (x, np.nanmedian(ys, axis=0))
    return data.curves[cumulative]


def draw_annthyroid(original, final):
    before, after = original.auc, final.auc
    pd.testing.assert_index_equal(before.index, after.index)
    lower = min(before.min(), after.min()) - 0.025
    upper = max(before.max(), after.max()) + 0.025
    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    ax.plot([lower, upper], [lower, upper], "--", color="#64748b", linewidth=1, label="Equal AUC")
    for mask, label, color in (
        (after > before, "Final higher", "#2563a6"),
        (after < before, "Original higher", "#c45324"),
        (after == before, "Tie", "#64748b"),
    ):
        if mask.any():
            ax.scatter(before[mask], after[mask], color=color, s=42, edgecolors="white",
                       linewidths=0.6, label=f"{label} ({mask.sum()})", zorder=3)
    ax.set(xlim=(lower, upper), ylim=(lower, upper), aspect="equal",
           xlabel="Original ROC AUC (depth limit 3)", ylabel="Final ROC AUC (depth limit 6)",
           title=f"Annthyroid: paired ROC AUC ({len(before)} seeds)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIGURES / "annthyroid_paired_auc.png", dpi=170)
    plt.close(fig)


def draw_timeline(models, drifts):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained", sharex=True, sharey=True)
    colors = {2048: "#64748b", 4096: "#2563a6", 16384: "#c45324"}
    for ax, cumulative in zip(axes, (True, False)):
        for window, data in models.items():
            x, y = median_curve(data, cumulative)
            if cumulative:
                np.testing.assert_allclose(y[-1], data.auc.median(), rtol=0, atol=ATOL)
            ax.plot(x, y, color=colors[window], label=f"Window {window:,}", linewidth=1.65)
        for i, drift in enumerate(drifts):
            ax.axvline(drift, color="#333333", linestyle=":", alpha=0.55, linewidth=0.85,
                       label="Drift marker" if i == 0 else None)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title("Cumulative AUC" if cumulative else "Centered-window AUC")
        ax.set_ylabel("ROC AUC")
    axes[-1].set_xlabel("Stream position (observations)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("INSECTS: median ROC AUC across seeds 0–13")
    fig.savefig(FIGURES / "insects_auc_timeline.png", dpi=170)
    plt.close(fig)


def draw_sweep(sweep):
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    windows = sorted(sweep)
    for seed in sweep[windows[0]].auc.index:
        ax.plot(windows, [sweep[w].auc.at[seed] for w in windows], color="#9baaba", alpha=0.8,
                linewidth=1, marker="o", markersize=4)
    ax.plot(windows, [sweep[w].auc.median() for w in windows], color="#205b96", marker="o",
            linewidth=2.2, label="Median across seeds")
    ax.axvline(16384, color="#c45324", linestyle="--", label="Final setting")
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows, labels=[f"{w:,}" for w in windows])
    ax.set(xlabel="Training window (observations)", ylabel="Full-stream ROC AUC",
           title="INSECTS: window-size sensitivity on selection seeds 5–7")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(FIGURES / "insects_window_sweep.png", dpi=170)
    plt.close(fig)


def segment_auc(data, drifts):
    """Split the global positive-negative pair sum into within/across drift regimes."""
    within_auc, across_auc = [], []
    for scores, labels in data.records.values():
        ends = [0, *drifts, len(labels)]
        within_weight = within_sum = 0.0
        for start, stop in zip(ends, ends[1:]):
            y = labels[start:stop]
            weight = y.sum() * (len(y) - y.sum())
            if weight:
                within_weight += weight
                within_sum += weight * roc_auc_score(y, scores[start:stop])
        p = labels.sum()
        total_weight = p * (len(labels) - p)
        total_sum = total_weight * roc_auc_score(labels, scores)
        within_auc.append(within_sum / within_weight)
        across_auc.append((total_sum - within_sum) / (total_weight - within_weight))
    return np.median(within_auc), np.median(across_auc), 1 - within_weight / total_weight


def compare_drift(base, final, drifts):
    x, y_base = median_curve(base)
    x_final, y_final = median_curve(final)
    np.testing.assert_array_equal(x, x_final)
    rows = []
    details = []
    boundaries = [0, *drifts, len(next(iter(base.records.values()))[0])]
    for i, drift in enumerate(drifts):
        pre = np.flatnonzero((x - WIDTH//2 >= boundaries[i]) & (x + WIDTH//2 <= drift))[-1]
        post = np.flatnonzero((x - WIDTH//2 >= drift) & (x + WIDTH//2 <= boundaries[i+2]))[0]
        delta = (y_final[post] - y_base[post]) - (y_final[pre] - y_base[pre])
        details.append(delta)
        rows.append(f"| {drift:,} | {x[pre]:,} / {x[post]:,} | {y_base[pre]:.4f} → {y_base[post]:.4f} | "
                    f"{y_final[pre]:.4f} → {y_final[post]:.4f} | {delta:+.4f} |")
    return "\n".join(rows), sum(d < 0 for d in details), drifts[int(np.argmin(details))], drifts[int(np.argmax(details))]


def csv_labels(path):
    frame = pd.read_csv(ROOT / path)
    labels = frame.iloc[:, -1].to_numpy()
    assert np.isin(labels, [0, 1]).all()
    return labels, frame.shape[1] - 2


def checks_and_costs(ann_orig, ann_final, orig, previous, final, ann_labels, insect_labels):
    specs = [
        ("ann_code", "Annthyroid code", ANN_PATH, 2048, "code", ann_labels, ann_orig),
        ("ann_paper", "Annthyroid paper", ANN_PATH, 2048, "paper", ann_labels, ann_final),
        ("insects_w2048", "INSECTS 2048", INSECTS_PATH, 2048, "code", insect_labels, orig),
        ("insects_w4096", "INSECTS 4096", INSECTS_PATH, 4096, "code", insect_labels, previous),
        ("insects_w8192", "INSECTS 8192", INSECTS_PATH, 8192, "code", insect_labels, None),
        ("insects_w16384", "INSECTS 16384", INSECTS_PATH, 16384, "code", insect_labels, final),
        ("insects_w32768", "INSECTS 32768", INSECTS_PATH, 32768, "code", insect_labels, None),
    ]
    costs = []
    repeats = []
    raw_records = {}
    for key, name, path, window, mode, labels, baseline in specs:
        file = OUT / "cost" / key
        raw = json.loads(file.with_suffix(".json").read_text())
        raw_records[key] = raw
        assert (raw["dataset"], raw["window"], raw["depth_limit"], raw["seed"]) == (path, window, mode, 0)
        assert raw["trees"] == raw["eta"] == 32 and raw["batch"] == 100
        assert raw["metric"] == "axisparallel"
        assert (raw["max_features"], raw["shuffle"]) == ((None, True) if path == ANN_PATH else (10, False))
        scores, observed_labels = inspect_record(file.with_suffix(".npz"), labels, raw["auc"], path == ANN_PATH)
        assert raw["n"] == len(labels) and raw["peak_rss_kib"] > 0
        np.testing.assert_allclose(raw["amortized_us_per_row"], raw["learn_score_s"] * 1e6 / len(labels))
        if baseline:
            prior_scores, prior_labels = baseline.records[0]
            np.testing.assert_array_equal(observed_labels, prior_labels)
            np.testing.assert_allclose(scores, prior_scores, rtol=0, atol=ATOL)
            np.testing.assert_allclose(raw["auc"], baseline.auc.at[0], rtol=0, atol=ATOL)
            if key in {"ann_paper", "insects_w16384"}:
                error = np.max(np.abs(scores - prior_scores))
                repeats.append(f"| {name} | {raw['auc']:.10f} | {baseline.auc.at[0]:.10f} | {error:.3g} |")
        costs.append(
            f"| {name} | {raw['learn_score_s']:.3f} | {raw['load_through_auc_s']:.3f} | "
            f"{raw['amortized_us_per_row']:.2f} | {raw['peak_rss_kib']/1024:.2f} |"
        )
    return "\n".join(costs), "\n".join(repeats), raw_records


def inventory():
    rows = []
    roots = [ARCHIVE, ANN_TUNING, INSECTS_TUNING, ATTACHMENTS, ROOT.parent / "review-inputs", OUT]
    for directory in roots:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".csv", ".npz", ".log", ".json", ".png", ".pdf"}:
                continue
            if path.is_relative_to(OUT) and path.name in {"input_inventory.csv", "verification.json"}:
                continue
            rows.append((str(path.relative_to(ROOT.parent)), path.suffix[1:], path.stat().st_size))
    return rows


def write_inventory():
    path = OUT / "input_inventory.csv"
    with path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["workspace_relative_path", "type", "bytes"])
        writer.writerows(inventory())


def sha256_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_assertion():
    # The report preserves the original experiment audit after the GitHub update.
    evidence = DOC / "experiment_evidence"
    initial = (evidence / "source-before.sha256").read_text().splitlines()
    for record in initial:
        expected, path = record.split(maxsplit=1)
        if path == "README.md":
            continue  # The repository landing page was updated after the experiments.
        assert sha256_file(ROOT / path) == expected, f"Experiment source changed: {path}"
    diff = (evidence / "diff-before.txt").read_text().rstrip()
    return diff, len(initial)


def build_report():
    ann_labels, ann_dim = csv_labels(ANN_PATH)
    insect_labels, insect_dim = csv_labels(INSECTS_PATH)
    uploaded = ARCHIVE / "uploaded"
    ann_original = group([prefix(uploaded, ANN)], range(30), ann_labels, uploaded_ann=True, shuffled=True)
    ann_paper = group([prefix(uploaded, ANN, "paper")], range(30), ann_labels, uploaded_ann=True, shuffled=True)
    historical = {
        2048: group([prefix(uploaded, INSECTS, "f10")], range(5), insect_labels,
                    scores_dir=ARCHIVE / "uploaded_base"),
        4096: group([prefix(uploaded, INSECTS, "w4096")], range(5), insect_labels,
                    scores_dir=ARCHIVE / "uploaded_long"),
        16384: group([prefix(INSECTS_TUNING, INSECTS, "reference_w16384")], range(5), insect_labels,
                     expected_window=16384),
    }
    heldout = {
        2048: group([prefix(OUT / "heldout", INSECTS, f"w2048_seed{i}") for i in range(8, 14)],
                    range(8, 14), insect_labels, expected_window=2048),
        **{w: group([prefix(INSECTS_TUNING, INSECTS, f"validation_w{w}")], range(8, 14), insect_labels,
                    expected_window=w) for w in (4096, 16384)},
    }
    sweep = {
        **{w: group([prefix(INSECTS_TUNING, INSECTS, f"selection_w{w}")], range(5, 8), insect_labels,
                    expected_window=w) for w in (4096, 8192, 16384)},
        32768: group([prefix(OUT / "sweep", INSECTS, f"w32768_seed{i}") for i in range(5, 8)],
                     range(5, 8), insect_labels, expected_window=32768),
    }
    original_fill = group(
        [prefix(OUT / "seed_fill", INSECTS, f"w2048_seed{i}") for i in range(5, 8)],
        range(5, 8), insect_labels, expected_window=2048,
    )
    middle = {2048: original_fill, 4096: sweep[4096], 16384: sweep[16384]}
    pooled = {}
    for window in historical:
        parts = (historical[window], middle[window], heldout[window])
        summary = pd.concat([part.summary for part in parts], ignore_index=True)
        assert summary.seed.tolist() == list(range(14)), window
        records = {seed: scores for part in parts for seed, scores in part.records.items()}
        pooled[window] = Group(summary, records)

    published = pd.read_csv(DOC / "experiment_evidence" / "paired_auc.csv")
    published = published[(published.dataset == "INSECTS") &
                          published.phase.isin(["reference", "selection", "validation", "baseline_fill"])]
    for window, data in pooled.items():
        scores = published[published.setting == f"window={window};depth=code"].set_index("seed").roc_auc.sort_index()
        pd.testing.assert_series_equal(scores, data.auc, check_names=False, atol=ATOL, rtol=0)
    final = json.loads((INSECTS_TUNING / "locked-selection.json").read_text())
    assert final["locked_before_validation"] and final["selected_window"] == 16384
    for observation in final["ranking"]:
        np.testing.assert_allclose(observation["median_auc"], sweep[observation["window"]].auc.median(), atol=ATOL, rtol=0)
    drifts = list(DRIFT_POINTS)
    assert drifts == sorted(set(drifts)) and 0 < drifts[0] < drifts[-1] < len(insect_labels)
    assert not set(heldout[16384].auc.index) & set(sweep[16384].auc.index)

    ann_multi = group([prefix(ARCHIVE, ANN, "mw2048-4096")], range(30), ann_labels,
                      shuffled=True, expected_window=(2048, 4096))
    insect_multi = group([prefix(ARCHIVE, INSECTS, "mw2048-4096")], range(5), insect_labels,
                         expected_window=(2048, 4096))
    ann_alt = {w: group([prefix(ANN_TUNING, ANN, f"validation_paper_w{w}")], range(35, 45), ann_labels,
                        shuffled=True, expected_window=w, expected_depth="paper") for w in (2048, 8192)}
    multi_comparisons = [
        ("Annthyroid 對原版", ann_original, ann_multi),
        ("Annthyroid 對 paper", ann_paper, ann_multi),
        ("INSECTS 對原版", historical[2048], insect_multi),
        ("INSECTS 對 4096", historical[4096], insect_multi),
    ]
    multi_pairs = pd.read_csv(DOC / "multiwindow_results.csv")
    assert len(multi_pairs) == len(multi_comparisons)
    for row, (_, before, after) in zip(multi_pairs.itertuples(), multi_comparisons):
        wins, losses, ties, p, _ = paired(before, after)
        assert (row.wins, row.losses, row.ties) == (wins, losses, ties)
        np.testing.assert_allclose([row.baseline_median, row.multiwindow_median, row.p_two_sided],
                                   [before.auc.median(), after.auc.median(), p], rtol=0, atol=ATOL)
    heldout_auc = pd.read_csv(DOC / "window_insects_validation.csv").set_index("seed")
    for window, col in [(4096, "auc_baseline"), (16384, "auc_candidate")]:
        pd.testing.assert_series_equal(heldout_auc[col], heldout[window].auc, check_names=False, atol=ATOL, rtol=0)
    heldout_local = pd.read_csv(DOC / "window_insects_windowed.csv").set_index("window")
    for window in (4096, 16384):
        _, y = median_curve(heldout[window])
        np.testing.assert_allclose([y.mean(), y.min()], heldout_local.loc[window].to_numpy(), atol=ATOL, rtol=0)
    uploaded_local = pd.read_csv(DOC / "multiwindow_windowed_results.csv")
    for i, data in enumerate([historical[2048], historical[4096], insect_multi]):
        _, y = median_curve(data)
        np.testing.assert_allclose([y.mean(), y.min()],
                                   uploaded_local.iloc[i][["mean_window_median", "min_window_median"]].to_numpy(float),
                                   atol=ATOL, rtol=0)

    contrasts = [
        ("Annthyroid 0–29，code → paper", ann_original, ann_paper),
        ("INSECTS 0–4，2048 → 4096", historical[2048], historical[4096]),
        ("INSECTS 0–4，4096 → 16384", historical[4096], historical[16384]),
        ("INSECTS 0–4，2048 → 16384", historical[2048], historical[16384]),
        ("INSECTS 8–13，2048 → 4096", heldout[2048], heldout[4096]),
        ("INSECTS 8–13，4096 → 16384", heldout[4096], heldout[16384]),
        ("INSECTS 8–13，2048 → 16384", heldout[2048], heldout[16384]),
    ]
    pair_rows = []
    for name, before, after in contrasts:
        wins, losses, ties, p, delta = paired(before, after)
        pair_rows.append(f"| {name} | {wins}/{len(delta)} ({100*wins/len(delta):.1f}%) | {losses} | {ties} | "
                         f"{delta.median():+.6f} | {p:.6g} |")
    main_rows = []
    insect_windows = (2048, 4096, 16384)
    ann_settings = ((4, 2048), (2, 2048), (2, 2048))
    insect_settings = tuple((4, window) for window in insect_windows)
    for name, groups, settings in [
        ("Annthyroid 0–29", (ann_original, ann_paper, ann_paper), ann_settings),
        ("INSECTS 0–13", tuple(pooled[w] for w in insect_windows), insect_settings),
    ]:
        medians = [group.auc.median() for group in groups]
        cells = " | ".join(
            f"深度上限 {log(window / 32, base):g}，視窗 {window}<br>AUC {median:.4f}"
            for (base, window), median in zip(settings, medians)
        )
        delta = medians[2] - medians[0]
        main_rows.append(f"| {name} | {cells} | {f'{delta:+.4f}' if delta else '0'} |")
    stat_rows = []
    for name, data in [
        ("Annthyroid 0–29 原版 code", ann_original),
        ("Annthyroid 0–29 既有較佳／最終 paper", ann_paper),
        *[(f"INSECTS {cohort} window {w}", data) for cohort, cohort_data in
          [("0–4", historical), ("8–13", heldout)] for w, data in cohort_data.items()],
    ]:
        stat_rows.append(f"| {name} " + summarize(data.auc))
    sweep_rows = [f"| {window:,} | {'本輪補跑' if window == 32768 else '既有選擇紀錄'} " + summarize(data.auc)
                  for window, data in sweep.items()]
    local_rows = []
    for cohort, models in [("0–4", historical), ("8–13", heldout)]:
        for w, data in models.items():
            _, y = median_curve(data)
            local_rows.append(f"| {cohort} | {w:,} | {y.mean():.6f} | {y.min():.6f} |")
    drift_table, worse_post, worst_drift, best_drift = compare_drift(heldout[4096], heldout[16384], drifts)
    segment_rows = []
    across_auc = {}
    within_auc = {}
    for w in (4096, 16384):
        within, across, proportion = segment_auc(heldout[w], drifts)
        across_auc[w] = across
        within_auc[w] = within
        segment_rows.append(f"| {w:,} | {within:.6f} | {across:.6f} | {proportion:.2%} |")

    claims = [
        ("Annthyroid 原版", "0.6721", ann_original),
        ("Annthyroid paper", "0.7458", ann_paper),
        ("INSECTS 原版", "0.7789", historical[2048]),
        ("INSECTS 4096", "0.8101", historical[4096]),
        ("INSECTS 16384", "0.8267", historical[16384]),
    ]
    claim_rows = []
    for label, claim, data in claims:
        actual = f"{data.auc.median():.4f}"
        status = "一致" if actual == claim else "原始逐點分數與摘要相符，先前引用來源未確認"
        claim_rows.append(f"| {label} | {claim} | {actual} | {status} |")

    ann_wins, _, _, _, ann_deltas = paired(ann_original, ann_paper)
    insect_wins, _, _, insect_p, insect_deltas = paired(heldout[4096], heldout[16384])
    values = {
        "ann_wins": f"{ann_wins}/{len(ann_deltas)}",
        "insect_wins": f"{insect_wins}/{len(insect_deltas)}",
        "insect_p": f"{insect_p:.6g}",
        "insect_heldout_delta": f"{heldout[16384].auc.median() - heldout[4096].auc.median():.4f}",
        "main_rows": "\n".join(main_rows),
        "stat_rows": "\n".join(stat_rows),
        "pair_rows": "\n".join(pair_rows),
        "sweep_rows": "\n".join(sweep_rows),
        "local_rows": "\n".join(local_rows),
        "segment_rows": "\n".join(segment_rows),
        "across_delta": f"{across_auc[16384] - across_auc[4096]:+.6f}",
        "within_delta": f"{within_auc[16384] - within_auc[4096]:+.6f}",
        "claim_rows": "\n".join(claim_rows),
        "drift_rows": drift_table,
        "drift_worse": f"{worse_post}/{len(drifts)}",
        "worst_drift": f"{worst_drift:,}",
        "best_drift": f"{best_drift:,}",
        "drift_positions": "、".join(f"{d:,}" for d in drifts),
        "eval_width": f"{WIDTH:,}",
        "eval_step": f"{STEP:,}",
        "ann_start": f"{ann_original.auc.median():.4f}",
        "ann_final": f"{ann_paper.auc.median():.4f}",
        "ann_delta": contrast(ann_original, ann_paper),
        "insect_start": f"{historical[2048].auc.median():.4f}",
        "insect_previous": f"{historical[4096].auc.median():.4f}",
        "insect_final": f"{historical[16384].auc.median():.4f}",
        "insect_delta": contrast(historical[4096], historical[16384]),
        "insect_heldout": f"{heldout[4096].auc.median():.4f} → {heldout[16384].auc.median():.4f}",
        "heldout_local": f"{heldout_local.at[4096, 'mean_window_median']:.4f} → {heldout_local.at[16384, 'mean_window_median']:.4f}",
        "pooled_local": " → ".join(f"{median_curve(pooled[w])[1].mean():.4f}" for w in (4096, 16384)),
        "original_local": f"{median_curve(historical[4096])[1].mean():.4f} → {median_curve(historical[16384])[1].mean():.4f}",
        "label_facts": (
            f"| Annthyroid | {len(ann_labels):,} | {ann_dim} | {ann_dim} | {ann_labels.sum():,} ({ann_labels.mean():.2%}) |\n"
            f"| INSECTS | {len(insect_labels):,} | {insect_dim} | 10 | {insect_labels.sum():,} ({insect_labels.mean():.2%}) |"
        ),
        "multi_rows": "\n".join(
            f"| {name} | {before.auc.median():.6f} | {after.auc.median():.6f} | "
            f"{paired(before, after)[0]}/{len(before.auc)} | {paired(before, after)[3]:.6g} |"
            for name, before, after in multi_comparisons
        ),
        "multi_local": f"{uploaded_local.iloc[0].mean_window_median:.4f} → {uploaded_local.iloc[2].mean_window_median:.4f}",
        "multi_long_local": f"{uploaded_local.iloc[1].mean_window_median:.4f}",
        "ann_alt": f"{ann_alt[2048].auc.median():.6f} → {ann_alt[8192].auc.median():.6f}",
        "ann_alt_p": f"{paired(ann_alt[2048], ann_alt[8192])[3]:.6g}",
        "ann_alt_wins": f"{paired(ann_alt[2048], ann_alt[8192])[0]}/{len(ann_alt[2048].auc)}",
        "sweep_peak": f"{max(sweep, key=lambda w: sweep[w].auc.median()):,}",
        "sweep_extra": f"{sweep[32768].auc.median():.6f}",
        "sweep_extra_wins": f"{(sweep[32768].auc > sweep[16384].auc).sum()}/{len(sweep[32768].auc)}",
        "ann_path": ANN_PATH,
        "insect_path": INSECTS_PATH,
        "verified_runs": str(sum(len(g.records) for g in [ann_original, ann_paper, ann_multi, insect_multi,
                            original_fill, *historical.values(), *heldout.values(), *sweep.values(), *ann_alt.values()])),
    }

    return values, pooled, sweep, drifts, (ann_original, ann_paper, *historical.values(), ann_labels, insect_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify report text and figure presence without rewriting")
    parser.add_argument("--inventory", action="store_true", help="Write the full artifact inventory and exit")
    args = parser.parse_args()
    if args.inventory:
        write_inventory()
        return
    values, pooled, sweep, drifts, originals = build_report()
    cost_rows, repeats, cost = checks_and_costs(*originals)
    source_diff, source_count = source_assertion()
    values.update(
        cost_rows=cost_rows,
        repeat_rows=repeats,
        source_diff=source_diff,
        source_count=source_count,
        cost_count=len(cost),
        cost_ratio=f"{cost['insects_w16384']['learn_score_s']/cost['insects_w4096']['learn_score_s']:.2f}",
        rss_delta=f"{(cost['insects_w16384']['peak_rss_kib']-cost['insects_w4096']['peak_rss_kib'])/1024:+.2f}",
    )
    template = (DOC / "experiment_report.template.md").read_text()
    report = re.sub(r"\{\{([a-z_]+)\}\}", lambda m: str(values[m[1]]), template)
    assert "{{" not in report
    file = DOC / "experiment_report.md"
    if args.check:
        assert file.read_text() == report, "Report is out of sync with artifacts"
    else:
        FIGURES.mkdir(exist_ok=True)
        draw_annthyroid(*originals[:2])
        draw_timeline(pooled, drifts)
        draw_sweep(sweep)
        write_inventory()
        file.write_text(report)
    figures = re.findall(r"\]\(\.\./figures/([^/]+\.png)\)", report)
    assert set(figures) == {"annthyroid_paired_auc.png", "insects_auc_timeline.png", "insects_window_sweep.png"}
    for name in figures:
        assert (FIGURES / name).stat().st_size > 4096
    verification = {"audited_npz": int(values["verified_runs"]) + len(cost),
                    "profiled_settings": len(cost), "figures": sorted(set(figures)),
                    "dataset_sha256": {
                        path: sha256_file(ROOT / path) for path in (ANN_PATH, INSECTS_PATH)
                    },
                    "figure_sha256": {
                        name: sha256_file(FIGURES / name) for name in sorted(set(figures))
                    },
                    "protected_files": source_count, "report_matches_sources": True}
    evidence = OUT / "verification.json"
    if args.check:
        assert verification == json.loads(evidence.read_text())
    else:
        evidence.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
