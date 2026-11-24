# 論文深度公式對照

兩組都用視窗 2048 和切分門檻 η＝32，比較原始程式的 log₄ 和論文的 log₂。滿視窗時對應的深度門檻為 3 和 6。

實驗使用 11 份靜態資料及 INSECTS。兩組都用 32 棵樹、每批約 100 筆，並用相同的種子 0–29 配對。靜態資料依種子打亂，INSECTS 保留原順序並使用前 10 個特徵。[資料來源](https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/e528c86b77219791d026d732f618efb0c7bde3c1/datasets)

## 結果

| 資料集 | 筆數 | 原版深度 3 AUC | 論文深度 6 AUC | AUC 差值 | 論文深度 6勝出 |
|---|---:|---:|---:|---:|---:|
| Donors | 619,326 | 0.7693 | 0.8137 | +0.0444 | 20/30 |
| HTTP | 567,498 | 0.9946 | 0.9924 | -0.0022 | 10/30 |
| Forest Cover | 286,048 | 0.8517 | 0.8562 | +0.0046 | 16/30 |
| Fraud | 284,807 | 0.9368 | 0.9417 | +0.0048 | 17/30 |
| Mulcross | 262,144 | 0.9935 | 0.9944 | +0.0009 | 17/30 |
| SMTP | 95,156 | 0.8564 | 0.8811 | +0.0247 | 18/30 |
| Shuttle | 49,097 | 0.9941 | 0.9931 | -0.0009 | 13/30 |
| Mammography | 11,183 | 0.8358 | 0.8671 | +0.0313 | 23/30 |
| NYC Taxi Shingle | 10,273 | 0.5680 | 0.5588 | -0.0092 | 12/30 |
| Annthyroid | 7,200 | 0.6721 | 0.7458 | +0.0737 | 26/30 |
| Satellite | 6,435 | 0.6491 | 0.6281 | -0.0210 | 11/30 |
| INSECTS | 212,514 | 0.7556 | 0.7758 | +0.0202 | 24/30 |

表中 AUC 是各設定 30 次的整體 ROC AUC 中位數，差值為 論文深度 6 減 原版深度 3。勝出次數用相同種子逐對比較。

以「過半數資料集提升，且差值中位數大於零」判斷整體改善。結果為 **8 份提升、4 份下降、0 份持平**，差值中位數為 **+0.0047**。每份資料等權平均的差值為 +0.0143。這個設定在多數資料集上提升 AUC，仍有資料集退步。

![各資料集的 AUC 差值](../figures/depth_benchmark.png)

圖中右側代表 AUC 提升，左側代表下降。每份資料跑 30 個種子，反映同一批資料在不同亂數下的結果。Annthyroid 30 次結果和 INSECTS 視窗比較，見[個別實驗](experiment_report.md)。

## 重跑

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/depth_benchmark.py --runs 30
python experiments/depth_benchmark_report.py --runs 30
```

[逐種子 AUC](experiment_evidence/depth_seed_auc.csv)
