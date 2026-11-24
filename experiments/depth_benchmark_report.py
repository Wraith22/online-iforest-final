"""Publish a short report and plot from the completed paired-depth benchmark."""

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from depth_benchmark import DATASETS, MODES, ROOT, SEEDS


def summarize(path):
    data = pd.read_csv(path)
    expected = pd.MultiIndex.from_product(
        [DATASETS, MODES, SEEDS], names=["dataset", "depth_limit", "seed"]
    )
    actual = data.set_index(expected.names).index
    if not actual.is_unique or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Expected one result for every dataset, depth mode and seed")
    if not data.auc.between(0, 1).all():
        raise ValueError("AUC scores must be between zero and one")

    rows = []
    for dataset in DATASETS:
        group = data[data.dataset == dataset]
        metadata = group[["rows", "features", "anomalies", "data_sha256"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"Inconsistent dataset across runs: {dataset}")
        scores = group.pivot(index="seed", columns="depth_limit", values="auc")
        original, paper = scores.code.median(), scores.paper.median()
        rows.append({
            "dataset": dataset,
            "samples": int(metadata.iloc[0]["rows"]),
            "original": original,
            "paper": paper,
            "delta": paper - original,
            "wins": int((scores.paper > scores.code).sum()),
        })
    return pd.DataFrame(rows)


def plot_changes(rows, path):
    chart = rows.sort_values("delta")
    colors = ["#2563a6" if delta > 0 else "#c45324" for delta in chart.delta]
    fig, ax = plt.subplots(figsize=(8.7, 5.4), layout="constrained")
    ax.barh(chart.dataset, chart.delta, color=colors)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set(title="Effect of a deeper limit across 12 datasets",
           xlabel="ROC AUC change (depth 6 − depth 3 at full window)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(rows, path):
    improved = int((rows.delta > 0).sum())
    declined = int((rows.delta < 0).sum())
    tied = len(rows) - improved - declined
    median_delta = rows.delta.median()
    if improved == len(rows):
        conclusion = "這 12 份資料的中位數都提升，仍需新資料確認。"
    elif improved > len(rows) / 2 and median_delta > 0:
        conclusion = "這套共用設定在多數資料集上提升 AUC，仍有資料集退步。"
    else:
        conclusion = "這套設定未達到共用改善的條件，不適合當成通用預設。"
    table = "\n".join(
        f"| {row.dataset} | {row.samples:,} | {row.original:.4f} | {row.paper:.4f} | {row.delta:+.4f} | {row.wins}/5 |"
        for row in rows.itertuples()
    )
    report = f"""# 共用深度設定實驗

我們在 11 份靜態資料及 INSECTS 測試同一設定。視窗固定 2048，只把深度公式從 log₄ 改為 log₂。視窗滿時，深度門檻由 3 改為 6，允許樹做更多層切分。兩組都用 32 棵樹、葉節點門檻 32、每批約 100 筆，並用相同的種子 0–4 配對。靜態資料依種子打亂，INSECTS 保留原順序並使用前 10 個特徵。[資料來源](https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/e528c86b77219791d026d732f618efb0c7bde3c1/datasets)

## 結果

| 資料集 | 筆數 | 原版深度 3 AUC | 論文深度 6 AUC | AUC 差值 | 論文深度勝出 |
|---|---:|---:|---:|---:|---:|
{table}

表中 AUC 是各設定 5 次的整體 ROC AUC 中位數，差值為論文深度減原版。勝出次數是用相同種子逐對比較。

以「過半數資料集提升，且差值中位數大於零」判斷整體改善。結果為 **{improved} 份提升、{declined} 份下降、{tied} 份持平**，差值中位數為 **{median_delta:+.4f}**。每份資料等權平均的差值為 {rows.delta.mean():+.4f}。{conclusion}

![各資料集的 AUC 差值](../figures/depth_benchmark.png)

圖中右側代表 AUC 提升，左側代表下降。每份資料只跑 5 個種子，反映同一批資料在不同亂數下的結果。Annthyroid 30 次結果和 INSECTS 視窗比較，見[個別實驗](experiment_report.md)。

## 重跑

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/depth_benchmark.py
python experiments/depth_benchmark_report.py
```

[逐種子 AUC](experiment_evidence/depth_seed_auc.csv)
"""
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/depth_benchmark/paired_auc.csv")
    args = parser.parse_args()
    rows = summarize(args.input)
    plot_changes(rows, ROOT / "figures/depth_benchmark.png")
    write_report(rows, ROOT / "docs/depth_benchmark.md")
    evidence = ROOT / "docs/experiment_evidence/depth_seed_auc.csv"
    if args.input.resolve() != evidence.resolve():
        shutil.copyfile(args.input, evidence)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
