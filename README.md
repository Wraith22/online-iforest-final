# Online Isolation Forest — Data Mining 期末報告

本專案復現並改良 Leveni et al., "Online Isolation Forest", ICML 2024。

- 論文原始碼：https://github.com/ineveLoppiliF/Online-Isolation-Forest （MIT License）
- 本專案的 `OnlineIForest/` 以原作者程式碼為基礎。第一個 commit 是未修改的原始碼，之後每一項修改各自一個 commit，可用 `git log` 與 `git show <commit>` 查看。

---

## 最終設定與結果

| 資料集 | 最終設定 | 原版整體 ROC AUC | 最終整體 ROC AUC |
|---|---|---:|---:|
| Annthyroid（30 個種子） | depth limit = paper | 0.6721 | **0.7458** |
| INSECTS（種子 0–4） | window = 16384 | 0.7789 | **0.8267** |

INSECTS 既有的 4096 視窗在種子 0–4 的中位數為 0.8101。另用未參與選擇的種子 8–13 驗證，16384 相較 4096 的整體 AUC 中位數為 **0.7884 → 0.8167**，6/6 勝。平均滑動視窗 AUC 為 0.8130 → 0.8106，同機單次學習加評分時間為 172.4 → 237.3 秒。

[完整實驗報告](docs/experiment_report.md) · [逐 seed AUC、成本與驗證紀錄](docs/experiment_evidence/README.md) · [INSECTS AUC 時間曲線](figures/insects_auc_timeline.png) · [視窗趨勢](figures/insects_window_sweep.png)

下載資料集後，在專案根目錄用以下指令重現最終設定。

```bash
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --runs 30 --tag final_paper
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --window 16384 --no-shuffle --max-features 10 --runs 5 --tag final_w16384
```

## 1. 環境安裝

需要 Python 3.10 以上與 git-lfs（原作者的資料集存在 git-lfs）。

```
git lfs install
git clone https://github.com/ineveLoppiliF/Online-Isolation-Forest.git
git clone https://github.com/Wraith22/online-iforest-final.git
cd online-iforest-final
python -m venv .venv
.venv\Scripts\activate            # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 資料集

資料集不放在本 repo。把原作者 repo 的 `datasets/` 整個複製成本專案的 `data/`：

```
xcopy /e /i ..\Online-Isolation-Forest\datasets data      # Windows
cp -r ../Online-Isolation-Forest/datasets data            # macOS/Linux
```

確認 `data\stationary\ifor\Annthyroid.csv` 約 365 KB。如果只有約 130 bytes，代表 git-lfs 沒有作用，到原作者 repo 執行 `git lfs pull`。

## 3. 專案結構

```
OnlineIForest/              原作者的演算法（含本專案的修改，見第 6 節）
experiments/
  run_experiment.py         主要評估腳本
  compare.py                兩組結果的成對比較（Wilcoxon signed-rank 檢定）
  windowed_auc.py           滑動視窗 AUC 曲線（對照論文 Figure 7）
  ablation_leaf.py          消融實驗：只改葉節點深度補償
  profile_report.py         單次時間、延遲與峰值 RSS 量測
  report_analysis.py        報告生成與原始分數查核
docs/                       完整報告、逐 seed 結果與驗證紀錄
figures/                    INSECTS AUC 與視窗趨勢
data/                       資料集（不上傳）
results/                    實驗輸出（不上傳）
```

## 4. 評估流程

與原作者的實驗腳本相同：

1. 每次執行前打亂資料（有時間順序的 INSECTS 不打亂）
2. 資料切成每批約 100 筆，依序餵入
3. 每一批先 `learn_batch` 再 `score_batch`
4. 整條資料流跑完後，以所有分數計算一個 ROC AUC
5. 第 i 次執行使用亂數種子 i，所以不同設定之間可以逐次配對比較

參數與論文 Table 3 相同：ω = 2048、η = 32、τ = 32、branching factor = 2。

## 5. 如何重現所有結果

以下指令都在專案根目錄執行。每次執行會輸出 `results/<資料集>_oifor_<tag>_summary.csv`（每次的 AUC 與時間）和 `_run<i>.npz`（逐點分數）。

### 5.1 復現論文（原始碼設定）

```
python experiments\run_experiment.py --dataset data\stationary\ifor\Annthyroid.csv --runs 30
python experiments\run_experiment.py --dataset data\stationary\ifor\Satellite.csv --runs 30
python experiments\run_experiment.py --dataset data\stationary\ifor\Mammography.csv --runs 30
python experiments\run_experiment.py --dataset data\stationary\ifor\Shuttle.csv --runs 30
python experiments\run_experiment.py --dataset data\non-stationary\kqt\insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --runs 5 --no-shuffle --max-features 10 --tag f10
python experiments\windowed_auc.py --runs "results\insects*_oifor_f10_run*.npz"
```

### 5.2 改良：深度上限依論文設定

```
python experiments\run_experiment.py --dataset data\stationary\ifor\Annthyroid.csv --runs 30 --depth-limit paper --tag paper
python experiments\compare.py results\Annthyroid_oifor_summary.csv results\Annthyroid_oifor_paper_summary.csv
```

其他資料集把檔名換掉即可。INSECTS 另加 `--no-shuffle --max-features 10 --runs 5`。

### 5.3 消融實驗

```
python experiments\ablation_leaf.py --dataset data\stationary\ifor\Annthyroid.csv --runs 30 --tag leaflog2
python experiments\compare.py results\Annthyroid_oifor_summary.csv results\Annthyroid_oifor_leaflog2_summary.csv
```

### 5.4 其他方向：斜切、重置與視窗大小

```
# 斜切
python experiments\run_experiment.py --dataset data\stationary\ifor\Annthyroid.csv --runs 30 --metric oblique --tag oblique
python experiments\run_experiment.py --dataset data\stationary\ifor\Annthyroid.csv --runs 30 --metric oblique_constrained --tag oblique_constrained

# drift：依既有 drift 標記測試重置模型
python experiments\run_experiment.py --dataset data\non-stationary\kqt\insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --runs 5 --no-shuffle --max-features 10 --tag oracle --reset-at 51400 74500 101000 142500 152000

# drift：視窗大小（512、1024、4096 各跑一次）
python experiments\run_experiment.py --dataset data\non-stationary\kqt\insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --runs 5 --no-shuffle --max-features 10 --tag w512 --window 512

# 畫對照圖
python experiments\windowed_auc.py --runs "results\insects*_oifor_f10_run*.npz" --compare "results\insects*_oifor_oracle_run*.npz" --labels 原版 oracle
```

## 6. 對原始碼的修改

所有修改都有開關，**預設值的行為與原作者完全相同**（已驗證：相同種子得到逐位相同的 AUC）。

| 參數 | 選項 | 說明 | 修改的檔案 |
|---|---|---|---|
| `depth_limit` | `code`（預設）/ `paper` | 樹深上限 δ。論文寫 log₂(ω/η) = 6，原始碼實際為 log₄(ω/η) = 3。`paper` 依論文計算 | tree、forest |
| `metric` | `axisparallel`（預設）/ `oblique` / `oblique_constrained` | 切割方向。`oblique` 為隨機斜向切割，各分量除以節點在該軸的範圍以保持尺度不變；`oblique_constrained` 另在節點分裂取樣時，只保留落在所有祖先切割線正確一側的點 | tree |
| `--windows` | 例如 `2048 4096`（預設停用） | 多視窗集成，將 `--trees` 的總樹數分配到各視窗，按樹數加權平均各森林正規化後的異常分數 | multiwindow、run_experiment |

`run_experiment.py` 的 `--reset-at` 不改演算法本身，只在指定位置丟棄整個模型重新開始，用於 oracle 實驗。

## 7. 結果

以下 AUC 為中位數；成對比較為 30 次（INSECTS 為 5 次）相同種子逐一比較，p 值為 Wilcoxon signed-rank 檢定。

### 7.1 復現

| 資料集 | 論文 oIFOR | 本專案 | 論文時間 | 本專案時間 |
|---|---|---|---|---|
| Annthyroid | 0.685 | 0.672 | 2.00s | 2.3s |
| Satellite | 0.651 | 0.649 | 3.74s | 2.5s |
| Mammography | 0.854 | 0.836 | 3.32s | 3.9s |
| Shuttle | 0.992 | 0.994 | 16.35s | 15.6s |

差距皆在 0.02 以內。INSECTS 的滑動視窗 AUC 曲線形狀與論文 Figure 7 一致。

### 7.2 改良：深度上限依論文設定

| 資料集 | 原始碼 δ | 論文 δ | 贏 / 輸 | p | 時間 |
|---|---|---|---|---|---|
| Annthyroid | 0.672 | **0.746** | 26 / 4 | < 0.001 | 2.3s → 2.7s |
| Mammography | 0.836 | **0.867** | 23 / 7 | < 0.001 | 約持平 |
| Satellite | 0.649 | 0.628 | 11 / 19 | 0.09 | — |
| Shuttle | 0.994 | 0.993 | 13 / 17 | 0.67 | 15.6s → 26.0s |
| INSECTS（視窗 AUC 平均） | 0.818 | 0.817 | — | — | 76s → 113s |

- 兩個資料集顯著提升，其餘沒有顯著差異。Mammography 的 0.867 高於論文 Table 4 所有方法（最高為 asdIFOR 的 0.855）。
- Satellite 有 32% 的異常點，違反「異常點很少」的前提；樹較深時，大量的異常點本身也會形成密集區域。
- 代價是執行時間增加，最多約 67%（Shuttle）。

**消融實驗**：在原始碼中，log 的底數影響深度上限與葉節點深度補償兩處。

| 資料集 | 原始碼 | 只改葉節點補償 | 只改深度上限 |
|---|---|---|---|
| Annthyroid | 0.672 | 0.685 | **0.746** |
| Mammography | 0.836 | 0.837（不顯著） | **0.867** |

提升主要來自深度上限。樹的平均最大深度從 4.0 增加到約 5.1–5.4。

### 7.3 負面結果：斜切

| 資料集 | 軸平行 | 斜切 | 斜切＋限制取樣 |
|---|---|---|---|
| Annthyroid | 0.672 | 0.644（9/21，p=0.03） | 0.665（11/19，p=0.02） |
| Satellite | 0.649 | 0.626（10/20，p=0.008） | 0.628（9/21，p=0.01） |
| Mammography | 0.836 | 0.829（11/19，p=0.21） | 0.821（12/18，p=0.13） |
| Shuttle | 0.994 | 0.991（7/23，p=0.004） | 未測 |

斜切在各資料集都較差。oIFOR 用軸平行的方框記錄節點範圍，斜切後方框會大於真實區域；但修正這一點（限制取樣）後仍然較差。推測這些資料集的異常點在單一特徵上就很極端，軸平行切割最有效。

### 7.4 drift 適應與視窗大小（INSECTS）

| 設定 | 整條 AUC | 視窗 AUC 平均 |
|---|---|---|
| 原版（ω = 2048） | 0.779 | 0.818 |
| Oracle：在 5 個 drift 位置重置 | 0.787 | 0.809 |
| ω = 512 | 0.770 | 0.809 |
| ω = 1024 | 0.778 | — |
| ω = 4096 | 0.810 | 0.823 |
| ω = 16384（最終） | 0.827 | 0.815 |

這組已知 drift 位置重置的實驗沒有提升視窗 AUC；縮小視窗也沒有幫助。不過，重置會丟掉歷史資料，並不是所有適應方法的性能上限，因此不能據此排除其他 drift 適應方法。ω = 4096 的整條 AUC 在 5 個種子上皆提升，但兩側精確 Wilcoxon p = 0.0625，且此視窗是看過結果後選出的，仍需要額外資料流或預先指定的種子驗證。

### 7.5 多視窗集成（新實驗）

實作與數據核對見 [改進方法與驗證紀錄](docs/improvements.md)。這是已測試的探索性對照，最終設定採用單一視窗。

本次測試的多視窗 2048+4096（各 16 棵）在 Annthyroid 的 30 次中位數 AUC 為 0.6783，在 INSECTS 的 5 次為 0.7881；均高於原版 2048 視窗，但仍分別落後於既有論文深度與單視窗 4096。INSECTS 的視窗 AUC 平均為 0.8166，未改善原版的 0.8178，因此尚無證據支持改成預設方法。

```bash
# 總共 32 棵樹：2048 與 4096 視窗各 16 棵；不需要標籤或已知 drift 位置
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --runs 5 --no-shuffle --max-features 10 --windows 2048 4096 --tag mw2048-4096
python experiments/compare.py results/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male_oifor_f10_summary.csv results/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male_oifor_mw2048-4096_summary.csv
python -m unittest discover -s tests -v
```

`--windows` 與 `--window` 互斥；若不指定 tag，會自動產生 `mw2048-4096`，以免覆蓋預設實驗。每次實驗另存 `_config.json` 記錄設定與評估順序；`_summary.csv` 和 `_run<i>.npz` 格式維持相容。

### 7.6 最少參數的推薦設定

以完整資料流 AUC 為目標，最後推薦每個資料集相較原版只改一個模型參數：

- **Annthyroid：`--depth-limit paper`**，window 保持 2048。此次再調 window 至 8192 的驗證只有 6/10 勝、p=0.375，不作為新的保守推薦。
- **INSECTS：`--window 16384`**，深度保持 code。先用種子 5–7 選視窗，再以未參與選擇的 8–13 驗證；相對既有較佳 window=4096，完整資料流 AUC 中位數 0.7884 → 0.8167、6/6 勝、雙側 p=0.03125（未校正）。但平均滑動視窗 AUC 略降，保留資料上限增加四倍，不能稱為全面優勝。

完整推薦指令與限制見 [單參數調整紀錄](docs/window_tuning.md)。這輪不啟用多視窗方法，不調整樹數或葉節點門檻，也不修改全域預設。

## 8. 其他復現發現

1. **資料規格不一致**：原作者提供的 Annthyroid.csv 有 7200 筆、異常比例 7.42%；論文 Table 2 寫 6832 筆、7.00%。
2. **INSECTS 特徵數**：論文寫 d = 33，原作者腳本只取前 10 個特徵（`test_kqt_datasets.py`）。本專案照腳本使用 10 個。
3. **執行之間變異很大**：Annthyroid 30 次的 AUC 範圍為 0.58–0.83，論文只報告中位數。比較方法時必須多次配對比較，只比單次結果不可靠。
4. **原版的遺忘不會縮小葉節點邊界**：`recursive_unlearn` 只減少計數，葉節點的 min/max 不會縮小。舊分布的樹結構因此會殘留，這可能是 oracle 重置反而較差的原因之一（未驗證）。
