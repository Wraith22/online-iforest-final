# Online Isolation Forest 期末專題

我們重現 Leveni 等人的 Online Isolation Forest（ICML 2024），再各改一個參數，觀察兩份資料的異常偵測效果。[實驗報告](docs/experiment_report.md) · [實驗數據](docs/experiment_evidence/README.md) · [論文原始碼](https://github.com/ineveLoppiliF/Online-Isolation-Forest)

表內為滿視窗時的深度上限、訓練視窗大小和整體 ROC AUC 中位數。

| 資料集 | 原始程式 | 先前較佳設定 | 最終設定 |
|---|---|---|---|
| Annthyroid，種子 0–29 | 深度上限 3，視窗 2048<br>AUC 0.6721 | 深度上限 6，視窗 2048<br>AUC 0.7458 | 深度上限 6，視窗 2048<br>AUC **0.7458** |
| INSECTS，種子 0–4 | 深度上限 3，視窗 2048<br>AUC 0.7789 | 深度上限 3.5，視窗 4096<br>AUC 0.8101 | 深度上限 4.5，視窗 16384<br>AUC **0.8267** |

原始公式在 2048 筆時算出 3，論文公式算出 6。INSECTS 沿用原始公式，視窗加大後深度上限也跟著變。Annthyroid 的先前較佳就是最終設定，AUC 差值為 0。INSECTS 用 4096 當驗證基準，16384 在種子 8–13 上 6/6 勝，整體 AUC 中位數由 **0.7884 升至 0.8167**，平均滑動視窗 AUC 略降，運算時間增加。

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
