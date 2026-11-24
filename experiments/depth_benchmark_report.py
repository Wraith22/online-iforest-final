"""Generate a report and plot from the completed benchmark."""

import argparse
from math import log
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from depth_benchmark import DATASETS, MODES, RESULT_DIRS, ROOT, SEEDS, window_comparison_name


def summarize(path, eta_comparison=None, *, candidate_window=None, runs=len(SEEDS)):
    data = pd.read_csv(path)
    if candidate_window is not None:
        parameter, settings = "window", (2048, candidate_window)
        if not data.depth_limit.eq("code").all():
            raise ValueError("Both windows must use the original depth formula")
    elif eta_comparison:
        parameter, settings = "eta_policy", ("fixed", "adaptive")
        if not data.depth_limit.eq("code").all():
            raise ValueError("Both eta policies must use the original depth formula")
        expected_scores = data.eta_policy if eta_comparison == "joint" else "fixed"
        if not data.score_eta.eq(expected_scores).all():
            raise ValueError("Unexpected score normalization for the selected comparison")
        span = "trimmed" if eta_comparison == "trimmed" else "full"
        expected_span = data.eta_policy.map({"fixed": "none", "adaptive": span})
        if not data.eta_span.eq(expected_span).all():
            raise ValueError("Unexpected feature span for the selected comparison")
        fixed = data.loc[data.eta_policy.eq("fixed"), ["eta_min", "eta_median", "eta_max"]]
        if not fixed.eq(32).all().all():
            raise ValueError("The baseline must use eta 32")
        if not (data.eta_min.gt(0) & data.eta_min.le(data.eta_median) & data.eta_median.le(data.eta_max)).all():
            raise ValueError("Invalid eta summary")
    else:
        parameter, settings = "depth_limit", MODES
    expected = pd.MultiIndex.from_product(
        [DATASETS, settings, range(runs)], names=["dataset", parameter, "seed"]
    )
    actual = data.set_index(expected.names).index
    if not actual.is_unique or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Expected one result for every dataset, setting and seed")
    if not data.auc.between(0, 1).all():
        raise ValueError("AUC scores must be between zero and one")

    rows = []
    for dataset in DATASETS:
        group = data[data.dataset == dataset]
        metadata = group[["rows", "features", "anomalies", "data_sha256"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"Inconsistent dataset across runs: {dataset}")
        scores = group.pivot(index="seed", columns=parameter, values="auc")
        baseline, candidate = (scores[setting] for setting in settings)
        row = {
            "dataset": dataset,
            "samples": int(metadata.iloc[0]["rows"]),
            "original": baseline.median(),
            "candidate": candidate.median(),
            "delta": candidate.median() - baseline.median(),
            "wins": int((candidate > baseline).sum()),
        }
        if eta_comparison:
            selected = group[group.eta_policy == "adaptive"]
            row["eta_min"] = int(selected.eta_min.min())
            row["eta_max"] = int(selected.eta_max.max())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_changes(rows, path, eta_comparison=None, *, candidate_window=None, runs=len(SEEDS)):
    chart = rows.sort_values("delta")
    colors = ["#2563a6" if delta > 0 else "#c45324" for delta in chart.delta]
    fig, ax = plt.subplots(figsize=(8.7, 5.4), layout="constrained")
    ax.barh(chart.dataset, chart.delta, color=colors)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    if candidate_window is not None:
        ax.set(title=f"Training window {candidate_window} across 12 datasets",
               xlabel=f"ROC AUC change (window {candidate_window} − window 2048)")
    elif eta_comparison:
        score_label = {"joint": "adaptive score scale", "split-only": "fixed score scale",
                       "trimmed": "trimmed feature range"}[eta_comparison]
        ax.set(title=f"Adaptive eta across 12 datasets ({score_label})",
               xlabel="ROC AUC change (adaptive eta − fixed eta 32)")
    else:
        ax.set(title="Effect of a deeper limit across 12 datasets",
               xlabel="ROC AUC change (depth 6 − depth 3 at full window)")
    if runs != len(SEEDS):
        ax.set_title(f"{ax.get_title()}\n{runs} paired seeds per dataset")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(rows, path, eta_comparison=None, *, candidate_window=None, runs=len(SEEDS)):
    improved = int((rows.delta > 0).sum())
    declined = int((rows.delta < 0).sum())
    tied = len(rows) - improved - declined
    median_delta = rows.delta.median()
    mean_delta = rows.delta.mean()
    criterion = "過半數資料集提升，且差值中位數大於零"
    meets_criterion = improved > len(rows) / 2 and median_delta > 0
    if candidate_window is not None:
        criterion = "過半數資料集提升，且等權平均差值大於零"
        meets_criterion = improved > len(rows) / 2 and mean_delta > 0
    if improved == len(rows):
        conclusion = "這 12 份資料的中位數都提升，仍需新資料確認。"
    elif meets_criterion:
        conclusion = "這個設定在多數資料集上提升 AUC，仍有資料集退步。"
    else:
        conclusion = "這個設定未達到多數資料集改善的條件。"
    variant_comparison = ""
    if candidate_window is not None:
        title = f"訓練視窗 2048 與 {candidate_window}"
        threshold = max(0, log(candidate_window / 32, 4))
        comparison = (f"把原始程式的訓練視窗從 2048 改為 {candidate_window}，切分門檻 η 固定 32。"
                      f"深度門檻仍用原始程式的 log₄ 公式，滿視窗時對應的數值為 3 與 {threshold:g}。"
                      f"模型每批更新後最多保留 {candidate_window} 筆。")
        if candidate_window == 4096:
            comparison += "先前 INSECTS 的整體 AUC 在 4096 時有提升，因此選它做跨資料集驗證。"
        original, candidate = "視窗 2048", f"視窗 {candidate_window}"
        related = "這裡比較整體 AUC。INSECTS 的滑動視窗 AUC 與 14 個種子的視窗實驗另有紀錄。[個別實驗](experiment_report.md)"
        command_option = f" --candidate-window {candidate_window}"
        artifact = window_comparison_name(candidate_window)
    elif eta_comparison:
        shared_method = r"""論文第 6 節把自動選擇 $\eta$ 列為未來工作。$\eta$ 是葉節點分裂所需樣本數的基準。我們用資料集中程度選擇門檻，兩組都採用視窗 2048 和原始程式的 log₄ 深度公式。原版固定 $\eta=32$，自動版先用 32，之後每累積約一個視窗的新資料，就用最近 2048 筆重新估計。[論文](https://proceedings.mlr.press/v235/leveni24a.html)"""
        if eta_comparison == "trimmed":
            title = "自動選擇切分門檻 η"
            estimator = r"以各特徵的四分位距除以第 5 到 95 百分位的範圍，再取中位數 $r$，令 $\eta=\max(8,\operatorname{round}(32\sqrt{1.8r}))$。跳過範圍為零的特徵。如果全數跳過，沿用 32。這樣可減少極端值把範圍拉大的影響。"
            related = "η 範圍包含暖機的 32。三版在同一批資料上評估。[論文深度對照](depth_benchmark.md)"
            artifact = "eta"
            evidence = path.parent / "experiment_evidence"
            joint = evidence / "eta_joint_seed_auc.csv"
            split = evidence / "eta_split_seed_auc.csv"
            if joint.is_file() and split.is_file():
                variant_comparison = "\n## 三版比較\n\n| 版本 | 提升資料集 | 等權平均 AUC 差值 |\n|---|---:|---:|\n"
                for label, results in (
                    ("[全距與自動評分尺度](eta_joint_benchmark.md)", summarize(joint, "joint")),
                    ("[全距與固定評分尺度](eta_split_benchmark.md)", summarize(split, "split-only")),
                    ("百分位範圍與固定評分尺度", rows),
                ):
                    variant_comparison += f"| {label} | {(results.delta > 0).sum()}/{len(results)} | {results.delta.mean():+.4f} |\n"
        else:
            estimator = r"對每個非常數特徵，計算四分位距除以全距，再取中位數 $r$，令 $\eta=\max(8,\operatorname{round}(32\sqrt{2r}))$。全為常數時沿用 32。"
            title, artifact = ("自動 η 初版實驗", "eta_joint") if eta_comparison == "joint" else ("自動 η 固定評分尺度實驗", "eta_split")
            related = "η 範圍包含暖機的 32。[百分位範圍版](eta_benchmark.md)"
        if eta_comparison == "joint":
            scoring = "學習、遺忘和評分都使用當時的 η，因此葉節點路徑長度修正與分數正規化會隨 η 改變。"
        else:
            scoring = "學習和遺忘使用自動 η，訓練的深度門檻把當時的 η 帶入原公式。評分用 32 計算葉節點路徑長度修正與分數正規化。"
        comparison = f"{shared_method}\n\n{estimator} 計算只用特徵。中央資料越集中，門檻越小。均勻分布時門檻接近 32。\n\n{scoring}"
        original, candidate = "固定 η=32", "自動 η"
        command_option = f" --adaptive-eta {eta_comparison}"
    else:
        title = "論文深度公式對照"
        comparison = "兩組都用視窗 2048 和切分門檻 η＝32，比較原始程式的 log₄ 和論文的 log₂。滿視窗時對應的深度門檻為 3 和 6。"
        original, candidate = "原版深度 3", "論文深度 6"
        related = "Annthyroid 30 次結果和 INSECTS 視窗比較，見[個別實驗](experiment_report.md)。"
        command_option = ""
        artifact = "depth"
    eta_column = " 自動 η 範圍 |" if eta_comparison else ""
    eta_alignment = "---:|" if eta_comparison else ""
    if runs != len(SEEDS):
        command_option += f" --runs {runs}"
    seed_text = f"0–{runs - 1}" if runs > 1 else "0"
    table = "\n".join(
        f"| {row.dataset} | {row.samples:,} | {row.original:.4f} | {row.candidate:.4f} | {row.delta:+.4f} | {row.wins}/{runs} |"
        + (f" {row.eta_min}–{row.eta_max} |" if eta_comparison else "")
        for row in rows.itertuples()
    )
    report = f"""# {title}

{comparison}

實驗使用 11 份靜態資料及 INSECTS。兩組都用 32 棵樹、每批約 100 筆，並用相同的種子 {seed_text} 配對。靜態資料依種子打亂，INSECTS 保留原順序並使用前 10 個特徵。[資料來源](https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/e528c86b77219791d026d732f618efb0c7bde3c1/datasets)

## 結果

| 資料集 | 筆數 | {original} AUC | {candidate} AUC | AUC 差值 | {candidate}勝出 |{eta_column}
|---|---:|---:|---:|---:|---:|{eta_alignment}
{table}

表中 AUC 是各設定 {runs} 次的整體 ROC AUC 中位數，差值為 {candidate} 減 {original}。勝出次數用相同種子逐對比較。

以「{criterion}」判斷整體改善。結果為 **{improved} 份提升、{declined} 份下降、{tied} 份持平**，差值中位數為 **{median_delta:+.4f}**。每份資料等權平均的差值為 {mean_delta:+.4f}。{conclusion}

![各資料集的 AUC 差值](../figures/{artifact}_benchmark.png)

圖中右側代表 AUC 提升，左側代表下降。每份資料{'只' if runs == len(SEEDS) else ''}跑 {runs} 個種子，反映同一批資料在不同亂數下的結果。{related}
{variant_comparison}
## 重跑

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/depth_benchmark.py{command_option}
python experiments/depth_benchmark_report.py{command_option}
```

[逐種子 AUC](experiment_evidence/{artifact}_seed_auc.csv)
"""
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--runs", type=int, default=len(SEEDS), help="Number of expected seeds starting at 0")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--adaptive-eta", choices=["joint", "split-only", "trimmed"])
    comparison.add_argument("--candidate-window", type=int)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.candidate_window is not None:
        if args.candidate_window < 1 or args.candidate_window == 2048:
            parser.error("--candidate-window must be positive and differ from 2048")
        artifact = results_dir = window_comparison_name(args.candidate_window)
    else:
        artifact = {None: "depth", "joint": "eta_joint", "split-only": "eta_split", "trimmed": "eta"}[args.adaptive_eta]
        results_dir = RESULT_DIRS[args.adaptive_eta]
    source = args.input or ROOT / f"results/{results_dir}_benchmark/paired_auc.csv"
    rows = summarize(source, args.adaptive_eta, candidate_window=args.candidate_window, runs=args.runs)
    plot_changes(rows, ROOT / f"figures/{artifact}_benchmark.png", args.adaptive_eta,
                 candidate_window=args.candidate_window, runs=args.runs)
    write_report(rows, ROOT / f"docs/{artifact}_benchmark.md", args.adaptive_eta,
                 candidate_window=args.candidate_window, runs=args.runs)
    evidence = ROOT / f"docs/experiment_evidence/{artifact}_seed_auc.csv"
    if source.resolve() != evidence.resolve():
        shutil.copyfile(source, evidence)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
