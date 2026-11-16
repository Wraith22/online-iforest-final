r"""
計算並畫出「滑動視窗 AUC」曲線，對照論文 Figure 7。

論文 Figure 7 的做法：在每個時間點 t，取以 t 為中心、寬 5000 筆的視窗內的分數計算 ROC AUC，
用來觀察模型在資料分布改變（drift）前後的「即時」表現。最後的單一 AUC 看不出這件事。

用法（在專案根目錄執行）：
    python experiments\windowed_auc.py --runs "results\insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male_oifor_f10_run*.npz"

可以用 --compare 再給第二組結果，兩條中位數曲線畫在同一張圖：
    python experiments\windowed_auc.py --runs "results\A_run*.npz" --compare "results\B_run*.npz" --labels 原版 改進版

輸出：
    與第一組結果同名的 _windowed.png（有 --compare 時為 _compare_windowed.png）（圖）與 _windowed.csv（每個中心點各次執行的 AUC）
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 只存檔，不跳視窗
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def windowed_auc(scores, labels, width, step):
    n = len(scores)
    half = width // 2
    centers = np.arange(half, n - half + 1, step)
    aucs = np.full(len(centers), np.nan)
    for k, c in enumerate(centers):
        y = labels[c - half:c + half]
        if 0 < y.sum() < len(y):  # 視窗內同時有正常與異常才算得出 AUC
            aucs[k] = roc_auc_score(y, scores[c - half:c + half])
    return centers, aucs


def load_group(pattern, width, step):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"找不到符合的檔案：{pattern}")
    curves = []
    for f in files:
        d = np.load(f)
        centers, aucs = windowed_auc(d["scores"], d["labels"], width, step)
        curves.append(aucs)
    return files, centers, np.vstack(curves)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="第一組 npz 檔（可用 * 萬用字元，記得加引號）")
    p.add_argument("--compare", default=None, help="第二組 npz 檔（選用）")
    p.add_argument("--labels", nargs="+", default=["oIFOR"], help="圖例名稱")
    p.add_argument("--width", type=int, default=5000, help="視窗寬度（論文用 5000）")
    p.add_argument("--step", type=int, default=1000, help="每隔幾筆算一次")
    args = p.parse_args()

    groups = [args.runs] + ([args.compare] if args.compare else [])
    labels = args.labels + [f"group{i}" for i in range(len(args.labels), len(groups))]

    fig, ax = plt.subplots(figsize=(10, 4))
    first_files, table = None, {}
    for g, (pattern, label) in enumerate(zip(groups, labels)):
        files, centers, curves = load_group(pattern, args.width, args.step)
        if first_files is None:
            first_files = files
            table["center"] = centers
        median = np.nanmedian(curves, axis=0)
        color = f"C{g}"
        for c in curves:
            ax.plot(centers, c, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(centers, median, color=color, linewidth=2,
                label=f"{label}（{len(files)} 次中位數，整體平均 {np.nanmean(median):.3f}）")
        for i, c in enumerate(curves):
            table[f"{label}_run{i}"] = c
        table[f"{label}_median"] = median
        print(f"{label}：{len(files)} 個檔案，視窗 AUC 中位數曲線的平均 = {np.nanmean(median):.4f}，"
              f"最低 = {np.nanmin(median):.4f}（在第 {int(centers[np.nanargmin(median)])} 筆附近）")

    ax.set_ylim(0, 1)
    ax.set_xlabel("資料流位置（第幾筆）")
    ax.set_ylabel(f"ROC AUC（寬 {args.width} 的滑動視窗）")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    stem = Path(first_files[0]).name.rsplit("_run", 1)[0] + ("_compare" if args.compare else "")
    out_dir = Path(first_files[0]).parent
    png, csv = out_dir / f"{stem}_windowed.png", out_dir / f"{stem}_windowed.csv"
    fig.savefig(png, dpi=150)
    pd.DataFrame(table).to_csv(csv, index=False)
    print(f"圖：{png}")
    print(f"數據：{csv}")


if __name__ == "__main__":
    main()
