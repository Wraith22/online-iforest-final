# Annthyroid：只調一個參數 window

## 固定條件與預先規則

基準為單視窗 `window=2048`、`depth_limit=paper`、32 棵樹、η=32、batch=100、axisparallel。維持打亂資料及 learn→score 流程。本次唯一調整的參數是 `window`；不改深度、樹數或分裂門檻。

選擇集種子固定為 30–34；候選只含 4096、8192。依完整資料流 AUC 的中位數挑唯一替代視窗（同分選較小值）。在讀取驗證結果前將決策寫入 `results/window_only_ann/selection_locked.json`。驗證種子為不重疊的 35–44。這只是同一資料集的新亂數種子，並非新資料集或嚴格的資料保留測試。

## 選擇集（每設定 5 次）

| window | AUC 中位數 |
|---|---:|
| 2048（基準） | 0.733460 |
| 4096 | 0.744345 |
| 8192 | 0.747338 |

鎖定唯一替代候選為 **8192**，未在驗證之後更換候選。

## 驗證集（每設定 10 次）

| 設定 | AUC 中位數 | 執行時間中位數（秒） |
|---|---:|---:|
| 基準 window=2048 | 0.777627 | 1.883 |
| 候選 window=8192 | 0.783376 | 1.702 |

同種子配對：候選 **6 勝／4 輸／0 平手**；配對 AUC 差的中位數 +0.001855，平均 +0.001983；雙側 Wilcoxon p=0.375000。這個 p 只針對事先鎖定的替代候選與基準比較，不能證明對其他資料集普遍改善。

## 重現與證據

所有 35 次輸出都保留在 `results/window_only_ann/`，包含參數 JSON、summary CSV、逐點 NPZ、命令日誌。已用 NPZ 的 scores／labels 重新計算每次 AUC，逐一比對摘要；全部 7200 筆分數皆有限，種子範圍與執行次數也已核對。逐次數值見 [CSV](window_ann_results.csv)。時間由同一雲端實例量測，但可能有其他實驗並行，不能視為嚴格效能基準。

```bash
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --window 2048 --seed 35 --runs 10 --tag validation_paper_w2048 --out results/window_only_ann
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --window 8192 --seed 35 --runs 10 --tag validation_paper_w8192 --out results/window_only_ann
```

## 建議

候選 8192 在新的驗證種子上有小幅數值提升（中位數 0.777627 → 0.783376），但只有 6 勝 4 輸、p=0.375，沒有足夠證據說它穩定優於既有的 paper+2048。建議將 **paper+2048 保留為保守設定**；8192 可列為待更多種子與新資料驗證的候選，不為追求漂亮數值強行切換預設。

8192 大於 Annthyroid 的全部 7200 筆資料，因此這次在該候選下不會有資料因視窗容量而被遺忘；這是有限靜態資料流上的結果，不能推論為長期串流的最佳記憶長度或 drift 適應能力。8192 也有較高的保留資料上限。相較專案原版的 code+2048，paper+8192 總共變更兩個參數（depth_limit、window）；本輪則是在已有的 paper+2048 較佳設定上，僅搜尋 window 一個參數。
