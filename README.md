# Online Isolation Forest 期末專題

我們重現 Leveni 等人的 Online Isolation Forest（ICML 2024），再各改一個參數，觀察兩份資料的異常偵測效果。[實驗報告](docs/experiment_report.md) · [實驗數據](docs/experiment_evidence/README.md) · [論文原始碼](https://github.com/ineveLoppiliF/Online-Isolation-Forest)

| 資料集 | 原版 | 最終設定 | 整體 ROC AUC 中位數 |
|---|---|---|---:|
| Annthyroid，30 個種子 | `code` 深度、視窗 2048 | `paper` 深度、視窗 2048 | 0.6721 → **0.7458** |
| INSECTS，種子 0–4 | 視窗 2048 | 視窗 16384 | 0.7789 → **0.8267** |

Annthyroid 的最終設定就是先前較佳的 `paper`，兩者差值為 0。INSECTS 先前較佳的 4096 視窗是事先指定的驗證基準。16384 在新種子上 6/6 勝，整體 AUC 由 **0.7884 升至 0.8167**，平均滑動視窗 AUC 略降，運算時間增加。

## 執行

用 Python 3.12 建立虛擬環境並安裝依賴。資料請用 Git LFS 下載[原作者的資料集](https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/main/datasets)，將 `datasets` 複製為本專案的 `data`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --runs 30 --tag final_paper
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --window 16384 --no-shuffle --max-features 10 --runs 5 --tag final_w16384
```

Windows 使用 `.venv\Scripts\activate`。輸出在 `results/`。其他歷史實驗見[補充紀錄](docs/improvements.md)。
