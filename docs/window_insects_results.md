# INSECTS：只調一個參數 window

固定原版深度 code、軸平行切割、32 棵樹、η=32、batch=100、不打亂、前 10 特徵。只搜尋 window。完整資料流有 212514 筆。

## 選擇階段

事前指定種子 5–7、候選 4096/8192/16384，依完整資料流 AUC 中位數選最高者（同分選較小視窗）。完成後在驗證開始前寫入 `results/window_only_insects/locked-selection.json`，唯一設定鎖定為 **16384**。

| window | 選擇集 AUC 中位數（3 次） |
|---|---:|
| 4096 | 0.784476 |
| 8192 | 0.786091 |
| 16384 | 0.807622 |

## 未參與選擇的種子驗證

固定種子 8–13、各六次。沒有根據驗證結果再次選其他視窗、特徵數或深度設定。

| 方法 | 完整資料流 AUC 中位數 |
|---|---:|
| 基準 window=4096 | 0.788354 |
| 固定候選 window=16384 | 0.816655 |

候選 6 勝／0 輸／0 平手；配對差中位數 +0.021320，雙側 Wilcoxon p=0.031250。這與舊種子 0–4 的基準 0.810056 不是同一組驗證，不能直接以不同種子的中位數相減來宣稱收益。

## 視窗 AUC（次要指標）

每個設定的六次執行，於 208 個中心點取寬度 5000、步長 1000 的視窗 AUC，跨種子取中位數，再對時間軸取平均。此指標不參與候選選擇。

| window | 時間軸上的視窗中位數平均 | 最低視窗中位數 |
|---|---:|---:|
| 4096 | 0.812954 | 0.637137 |
| 16384 | 0.810634 | 0.637117 |

圖位於 `results/window_only_insects/validation_windows.png`。重疊視窗不是 208 個獨立實驗，不當作獨立樣本做顯著性檢定。

## 成本與解讀

本輪優先指標是完整資料流 AUC，據此推薦 **window=16384**；但平均滑動視窗 AUC 稍低於 4096，不能將完整資料流的收益解讀為局部 AUC 或 drift 適應速度也提升。若以 Figure 7 類型的即時／局部表現為主要目標，保留 4096 是合理的選擇。p=0.03125 是單一資料流、六個初始化種子的未校正結果，沒有證明跨資料流普遍有效。

16384 的保留資料上限為 4096 的四倍，雖維持總樹數 32，並不表示成本相同。並行實驗的時間僅供記錄，沒有據此宣稱加速或省記憶體。相同串流上的新亂數種子只能反映對初始化的穩定性，不能證明對未見資料流泛化。評估仍為 learn→score，不是嚴格的先評分再學習。

計畫、選擇鎖定、完整 CSV／NPZ／設定與日誌保留於 `results/window_only_insects/`。每次已驗證 NPZ 筆數、有限分數與重算 AUC，並核對種子和實際完成次數。逐次數值見 [全部 CSV](window_insects_runs.csv)、[驗證配對 CSV](window_insects_validation.csv)、[視窗結果 CSV](window_insects_windowed.csv)。

```bash
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --window 16384 --no-shuffle --max-features 10 --seed 8 --runs 6 --tag validation_w16384 --out results/window_only_insects
```

## 原先種子 0–4 的固定設定重現

在選擇已鎖定後，另用原先種子重跑 window=16384（不參與參數選擇）。相同種子的既有 window=4096 中位數為 0.810056，新設定為 0.826725；5 勝／0 輸，雙側 p=0.062500。這組重現不作獨立驗證；主要驗證仍是先鎖設定後跑的種子 8–13。配對表見 [CSV](window_insects_reference.csv)。

本輪 INSECTS 共完成 26 次（選擇 9、驗證 12、原種子重現 5），加上 Annthyroid 的 35 次，共 61 次；均保留原始輸出。
