# 實驗結果與驗證摘要

[回到完整實驗報告](../experiment_report.md)

| 檔案 | 內容 |
|---|---|
| [eta_calibration_auc.csv](eta_calibration_auc.csv) | 7 份資料的逐種子 AUC、選出的 η 與兩類模擬測試結果。[校準報告](../eta_calibration.md) |
| [eta_seed_auc.csv](eta_seed_auc.csv) | 以百分位範圍估計 η，記錄 12 份資料的逐種子 AUC 和 η 範圍。[實驗報告](../eta_benchmark.md) |
| [eta_split_seed_auc.csv](eta_split_seed_auc.csv) | 比較以全距估計 η 並固定評分尺度的版本。[實驗報告](../eta_split_benchmark.md) |
| [eta_joint_seed_auc.csv](eta_joint_seed_auc.csv) | 初版讓訓練和評分都使用自動 η。[初版結果](../eta_joint_benchmark.md) |
| [depth_seed_auc.csv](depth_seed_auc.csv) | 12 份資料的原始與論文深度公式比較，每份資料 5 個配對種子。[論文對照](../depth_benchmark.md) |
| [paired_auc.csv](paired_auc.csv) | 163 筆逐 seed ROC AUC。project_summary 欄記錄原始摘要在實驗工作區的相對路徑 |
| [INSECTS 選擇計畫](plan.json)及[鎖定紀錄](locked-selection.json) | 候選視窗與使用的 seeds，驗證前鎖定 16384 |
| [INSECTS 驗證比較](validation-result.json) | 4096 與 16384 在六個驗證 seeds 的配對結果 |
| [成本量測](cost/) | 各設定的 seed 0 AUC、執行時間、每筆攤銷時間與峰值 RSS |
| [自動驗證摘要](verification.json)及[來源檔雜湊](source-before.sha256)及[當時的 diff 摘要](diff-before.txt) | 實驗工作區的重新計算、圖與資料雜湊，以及模型檔案盤點 |

逐 seed CSV 的 phase 區分歷史對照、參數選擇、驗證、Annthyroid 的補充驗證與多視窗集成。baseline_fill 是原版為補齊種子 0–13 而補跑的三次。INSECTS 主表用 dataset、setting 和 seed 對齊三組，共各 14 筆。

完整 CSV 資料集、逐點 NPZ 和實驗日誌留在實驗工作區。`experiments/report_analysis.py` 的原始分數查核與重繪需要該工作區的原始輸入。從 GitHub 取得的版本可直接閱讀報告、圖與摘要，或按報告的指令使用原始資料重跑指定設定。
