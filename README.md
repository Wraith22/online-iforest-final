# Online Isolation Forest 期末專題

本專題重現 Leveni 等人的 Online Isolation Forest（ICML 2024）。深度與訓練視窗的比較都涵蓋 12 份資料，各用 30 個相同種子配對。[共用視窗實驗](docs/window_benchmark.md) · [論文深度對照](docs/depth_benchmark.md) · [自動 η 實驗](docs/eta_benchmark.md) · [個別資料集報告](docs/experiment_report.md) · [實驗數據](docs/experiment_evidence/README.md) · [論文原始碼](https://github.com/ineveLoppiliF/Online-Isolation-Forest)

用模擬異常選擇 η 的實驗涵蓋 7 份資料。[校準方法與結果](docs/eta_calibration.md)

## 個別資料集設定

Annthyroid 用 30 個種子重現論文的深度公式。視窗固定 2048，滿視窗時的深度門檻由原始程式的 3 變為論文的 6，整體 ROC AUC 中位數由 **0.6721 提高至 0.7458**。

INSECTS 用相同的 14 個種子比較視窗。整體 ROC AUC 中位數由 **0.7616**（2048）提高至 **0.7951**（4096），再到 **0.8185**（16384）。

## 執行

用 Python 3.12 建立虛擬環境並安裝依賴。資料請用 Git LFS 下載[原作者的資料集](https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/main/datasets)，將 `datasets` 複製為本專案的 `data`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/depth_benchmark.py --runs 30
python experiments/depth_benchmark.py --runs 30 --candidate-window 4096
python experiments/depth_benchmark.py --adaptive-eta trimmed
python experiments/run_experiment.py --dataset data/stationary/ifor/Annthyroid.csv --depth-limit paper --runs 30 --tag final_paper
python experiments/run_experiment.py --dataset data/non-stationary/kqt/insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male.csv --window 16384 --no-shuffle --max-features 10 --runs 14 --tag final_w16384
```

Linux 安裝 `g++` 後，可在兩組 30 種子的指令加上 `--native` 加速。Windows 使用 `.venv\Scripts\activate`。輸出在 `results/`。其他歷史實驗見[補充紀錄](docs/improvements.md)。
