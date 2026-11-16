r"""
成對比較兩組實驗結果（同一組種子），判斷差異是不是運氣。

用法（在專案根目錄執行）：
    python experiments\compare.py results\Satellite_oifor_summary.csv results\Satellite_oifor_oblique_summary.csv

會印出：兩邊的中位數 AUC、逐一配對後的差距、改進版贏了幾次，
以及 Wilcoxon signed-rank 檢定的 p 值（p < 0.05 才算統計上顯著）。
"""
import sys

import pandas as pd
from scipy.stats import wilcoxon


def main():
    if len(sys.argv) != 3:
        raise SystemExit("用法：python experiments\\compare.py 原版_summary.csv 改進版_summary.csv")
    a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    m = a.merge(b, on="seed", suffixes=("_a", "_b"))
    if len(m) == 0:
        raise SystemExit("兩個檔案沒有相同的種子，無法配對比較")
    diff = m.auc_b - m.auc_a
    print(f"配對數：{len(m)}")
    print(f"A 中位數 AUC = {m.auc_a.median():.4f}    時間 {m.time_s_a.median():.1f}s")
    print(f"B 中位數 AUC = {m.auc_b.median():.4f}    時間 {m.time_s_b.median():.1f}s")
    print(f"B − A：中位數 {diff.median():+.4f}，平均 {diff.mean():+.4f}")
    print(f"B 贏 {(diff > 0).sum()} 次，輸 {(diff < 0).sum()} 次，平手 {(diff == 0).sum()} 次")
    if len(m) >= 6 and (diff != 0).any():
        p = wilcoxon(m.auc_b, m.auc_a).pvalue
        print(f"Wilcoxon p = {p:.4f}  →  {'顯著（p < 0.05）' if p < 0.05 else '不顯著，差異可能只是隨機'}")
    else:
        print("配對數太少（少於 6），不做檢定")


if __name__ == "__main__":
    main()
