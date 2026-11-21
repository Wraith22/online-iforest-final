# 只調一個參數：window

## 最終建議

以完整資料流 ROC AUC 為優先，每個資料集相較原版只需改一個模型參數：

| 資料集 | 推薦設定 | 相較原版改動 | 本輪驗證結論 |
|---|---|---|---|
| Annthyroid | `depth_limit=paper`，window 保持 2048 | 只改深度設定 | 再將 window 調到 8192 僅 6/10 勝、p=0.375，保守保留既有較佳設定 |
| INSECTS | `window=16384`，深度保持 code | 只改視窗 | 未參與選擇的六個種子，全流 AUC 中位數 0.788354 → 0.816655；6/6 勝，雙側 p=0.03125（未校正） |

INSECTS 的基準是既有較佳的 4096，而非較弱的原版 2048。改善針對完整資料流 AUC：驗證的平均滑動視窗 AUC 由 0.812954 略降至 0.810634，所以不能宣稱全面優於 4096。若研究目標是局部／即時 AUC，可保留 4096。16384 的資料保留上限也是 4096 的四倍；本輪不聲稱節省記憶體或加速。

Annthyroid 的推薦以既有 30 次配對結果支持：原版 code+2048 的中位數 0.672091，paper+2048 為 0.745782。這與本輪新驗證種子的中位數不同，不混用兩組種子計算收益。

```bash
# Annthyroid：唯一改動的模型參數為 depth-limit
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --runs 30 --tag recommended_paper

# INSECTS：唯一改動的模型參數為 window；資料處理沿用既有設定
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --window 16384 --no-shuffle --max-features 10 --seed 8 --runs 6 --tag recommended_w16384
```

`runs`、`seed`、`tag` 是實驗控制，不是這輪搜尋的模型參數。推薦不需要多視窗集成、不增加樹數，也不修改其他資料集的全域預設。CLI 已支援這些設定，因此本輪不需要更改演算法程式。

詳見 [Annthyroid 的選擇與驗證](window_ann_results.md)、[INSECTS 的選擇與驗證](window_insects_results.md)。

## 事前範圍與方法

本輪以既有較佳單視窗設定為基準，不再使用多視窗集成。唯一搜尋的演算法參數是 `window`。總樹數 32、葉節點門檻 32、批次大小 100、軸平行切割均固定。

| 資料集 | 固定深度設定 | 基準 window | 選擇階段候選 | 選擇種子 | 驗證種子 |
|---|---|---:|---|---|---|
| Annthyroid | paper | 2048 | 2048 / 4096 / 8192 | 30–34 | 35–44 |
| INSECTS | code | 4096 | 4096 / 8192 / 16384 | 5–7 | 8–13 |

Annthyroid 保留既有的打亂流程；INSECTS 固定不打亂，取前 10 個特徵。這些是原本的評估設定，沒有在本輪搜尋。

每個資料集都先完成選擇階段，再以完整資料流 AUC 中位數選出設定、寫入鎖定紀錄，才開始跑另一組種子驗證。驗證不通過時保留基準，不根據驗證集重新挑別的視窗或新增參數。視窗 AUC 為次要診斷指標，不用來事後改選。

這是相同資料流上的新亂數種子驗證，可衡量對初始化／排列的穩定性，不是未見資料集上的泛化測試。評估順序仍為專案的 learn → score，以便與既有結果比較。

Annthyroid 的固定 `paper` 深度是沿用既有較佳設定。本輪相對該設定只調 `window`；若從最原始設定計算，Annthyroid 最多涉及 `depth_limit` 與 `window` 兩個參數，INSECTS 只有 `window` 一個。

完整輸出、逐點分數與設定分別在 `results/window_only_ann/`、`results/window_only_insects/`，不覆蓋先前結果。原先多視窗實作保留為可選實驗，但不參與本輪搜尋。
