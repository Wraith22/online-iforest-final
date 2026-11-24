# Online Isolation Forest 期末專題

本專題重現 Leveni 等人的 Online Isolation Forest（ICML 2024），並在 12 份資料上比較同一個深度設定。[12 資料集實驗](docs/depth_benchmark.md) · [Annthyroid 與 INSECTS 報告](docs/experiment_report.md) · [實驗數據](docs/experiment_evidence/README.md) · [論文原始碼](https://github.com/ineveLoppiliF/Online-Isolation-Forest)

## 個別資料集設定

表內為滿視窗時的深度上限、訓練視窗大小和整體 ROC AUC 中位數。

| 資料集 | 原始程式 | 先前較佳設定 | 最終設定 |
|---|---|---|---|
| Annthyroid，種子 0–29 | 深度上限 3，視窗 2048<br>AUC 0.6721 | 深度上限 6，視窗 2048<br>AUC 0.7458 | 深度上限 6，視窗 2048<br>AUC **0.7458** |
| INSECTS，種子 0–13 | 深度上限 3，視窗 2048<br>AUC 0.7616 | 深度上限 3.5，視窗 4096<br>AUC 0.7951 | 深度上限 4.5，視窗 16384<br>AUC **0.8185** |

Annthyroid 只改深度公式。INSECTS 只改視窗，三組都用相同的 14 個種子比較。

## 執行

用 Python 3.12 建立虛擬環境並安裝依賴。資料請用 Git LFS 下載[原作者的資料集](https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/main/datasets)，將 `datasets` 複製為本專案的 `data`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --runs 30 --tag final_paper
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --window 16384 --no-shuffle --max-features 10 --runs 14 --tag final_w16384
```

Windows 使用 `.venv\Scripts\activate`。輸出在 `results/`。其他歷史實驗見[補充紀錄](docs/improvements.md)。
